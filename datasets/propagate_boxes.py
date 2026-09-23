#!/usr/bin/env python3
"""
propagate_boxes.py

Expand box annotations by propagating each labeled frame's boxes to its
temporal neighbors, writing a drop-in box directory next to the original one:

    <root>/roboflow_boxes/boxes_<id>.pkl                  (read)
    <root>/roboflow_boxes_with_tracking/boxes_<id>.pkl    (written)

Seeds are the '<root> <img_id>' lines of --samples (same list format as
MultiSpectralDetection). Neighbors are img_id +/- k in <root>/color, chained
frame to frame with DIS optical flow. Each box moves by the median flow of its
central region, then is flowed back; it is kept only if the round trip lands
within --fb_iou of where it started, and stops propagating once it fails.

Output pickles keep the loader's format, {class name: [(score, [x0, y0, x1, y1])]},
in full colour-frame coordinates. Originals keep score 1.0; propagated boxes
store the lowest round-trip IoU along their chain as the score (the loader
discards it, but it is there for filtering later).

For every root in --samples, all original pickles are copied into the new
directory, so it can replace box_dir directly. A frame that has an original
pickle is never overwritten, whether or not it is in --samples. When a frame
is reachable from several seeds, the nearest seed wins.
"""
import argparse
import os
import pickle
import shutil
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Propagate labeled boxes to neighboring frames via optical flow")
    p.add_argument("--samples", type=str, required=True,
                   help="list file, '<root> <img_id>' per line; these frames seed the propagation")
    p.add_argument("--box_dir", type=str, default="roboflow_boxes",
                   help="original box directory inside each root")
    p.add_argument("--out_box_dir", type=str, default="roboflow_boxes_with_tracking",
                   help="output box directory created inside each root")
    p.add_argument("--color_dir", type=str, default="color",
                   help="colour image directory inside each root")
    p.add_argument("--box_prefix", type=str, default="boxes_",
                   help="box pickle filename prefix")
    p.add_argument("--ext", type=str, default=".png",
                   help="colour image extension")
    p.add_argument("--radius", type=int, default=3,
                   help="max frames to propagate forward and backward from each seed")
    p.add_argument("--fb_iou", type=float, default=0.7,
                   help="min forward-backward round-trip IoU to keep a propagated box")
    p.add_argument("--min_inside", type=float, default=0.8,
                   help="min fraction of a propagated box that must remain inside the image")
    p.add_argument("--core_frac", type=float, default=0.5,
                   help="fraction of box width/height (centred) used to estimate the box's flow")
    p.add_argument("--flow_scale", type=float, default=1.0,
                   help="resize factor applied before flow computation (<1 is faster, less precise)")
    p.add_argument("--viz_dir", type=str, default="",
                   help="if set, write box overlays of propagated frames here for spot-checking")
    return p.parse_args()


# ---------------------------------------------------------------- io

def read_samples(path):
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                out.append((parts[0], parts[1]))
    return out


def load_boxes(path):
    """-> list of (name, [x0, y0, x1, y1]); same tolerance as the loader."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    out = []
    for name, entries in data.items():
        for entry in entries:
            box = entry[1] if isinstance(entry, (tuple, list)) and len(entry) == 2 else entry
            out.append((name, [float(v) for v in np.asarray(box).reshape(4)]))
    return out


def neighbor_id(img_id, k):
    n = int(img_id) + k
    return f"{n:0{len(img_id)}d}" if n >= 0 else None


# ---------------------------------------------------------------- geometry (xyxy)

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_frac(b, W, H):
    iw = max(0.0, min(b[2], W) - max(b[0], 0))
    ih = max(0.0, min(b[3], H) - max(b[1], 0))
    return iw * ih / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)


def clip_box(b, W, H):
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])]


def shift_box(b, flow, core_frac):
    """Translate box by the median flow over its central core region."""
    H, W = flow.shape[:2]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw = max(1.0, core_frac * (b[2] - b[0]) / 2)
    hh = max(1.0, core_frac * (b[3] - b[1]) / 2)
    x0, x1 = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
    y0, y1 = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    dx, dy = np.median(flow[y0:y1, x0:x1].reshape(-1, 2), axis=0)
    return [b[0] + float(dx), b[1] + float(dy), b[2] + float(dx), b[3] + float(dy)]


# ---------------------------------------------------------------- flow + propagation

class Flow:
    def __init__(self, scale):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.scale = scale

    def __call__(self, a, b):
        if self.scale == 1.0:
            return self.dis.calc(a, b, None)
        a2 = cv2.resize(a, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(b, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        f = self.dis.calc(a2, b2, None)
        return cv2.resize(f, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) / self.scale


def propagate(color_path, img_id, boxes, flow_fn, args):
    """boxes: [(name, xyxy)]. Returns {nbr_id: (offset, [(name, score, xyxy)])}."""
    out = {}
    first = cv2.imread(color_path(img_id), cv2.IMREAD_GRAYSCALE)
    if first is None:
        print(f"[warn] missing colour frame {color_path(img_id)}")
        return out
    for step in (1, -1):
        cur = [(name, 1.0, b) for name, b in boxes]
        prev = first
        for k in range(1, args.radius + 1):
            nid = neighbor_id(img_id, step * k)
            nxt = cv2.imread(color_path(nid), cv2.IMREAD_GRAYSCALE) if nid else None
            if nxt is None or not cur:
                break
            fwd, bwd = flow_fn(prev, nxt), flow_fn(nxt, prev)
            H, W = nxt.shape
            kept = []
            for name, score, b in cur:
                nb = shift_box(b, fwd, args.core_frac)
                if nb is None or inside_frac(nb, W, H) < args.min_inside:
                    continue
                back = shift_box(nb, bwd, args.core_frac)
                rt = iou(back, b) if back is not None else 0.0
                if rt < args.fb_iou:
                    continue
                kept.append((name, min(score, rt), clip_box(nb, W, H)))
            if kept:
                out[nid] = (step * k, kept)
            cur, prev = kept, nxt
    return out


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    samples = read_samples(args.samples)
    seeds_by_root = defaultdict(list)
    for root, img_id in samples:
        seeds_by_root[root].append(img_id)
    print(f"{len(samples)} seed frames in {len(seeds_by_root)} roots")

    flow_fn = Flow(args.flow_scale)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    total_frames = total_boxes = 0

    for root, img_ids in sorted(seeds_by_root.items()):
        in_dir = os.path.join(root, args.box_dir)
        out_dir = os.path.join(root, args.out_box_dir)
        os.makedirs(out_dir, exist_ok=True)

        def color_path(i):
            return os.path.join(root, args.color_dir, f"color_{i}{args.ext}")

        def box_name(i):
            return f"{args.box_prefix}{i}.pkl"

        # originals: copied as-is, and never overwritten by propagation
        originals = {f for f in os.listdir(in_dir) if f.startswith(args.box_prefix) and f.endswith(".pkl")}
        for f in originals:
            shutil.copy2(os.path.join(in_dir, f), os.path.join(out_dir, f))

        targets = {}   # nbr_id -> (|offset|, [(name, score, xyxy)])
        for img_id in img_ids:
            src = os.path.join(in_dir, box_name(img_id))
            if not os.path.exists(src):
                print(f"[warn] no boxes for listed frame {src}")
                continue
            boxes = load_boxes(src)
            if not boxes:
                continue
            for nid, (off, kept) in propagate(color_path, img_id, boxes, flow_fn, args).items():
                if box_name(nid) in originals:
                    continue
                if nid not in targets or abs(off) < targets[nid][0]:
                    targets[nid] = (abs(off), kept)

        for nid, (_, kept) in sorted(targets.items()):
            data = defaultdict(list)
            for name, score, b in kept:
                data[name].append((round(score, 3), [round(v, 2) for v in b]))
            with open(os.path.join(out_dir, box_name(nid)), "wb") as fh:
                pickle.dump(dict(data), fh)
            total_boxes += len(kept)
            if args.viz_dir:
                img = cv2.imread(color_path(nid))
                for _, _, b in kept:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
                tag = os.path.relpath(root, "/").replace(os.sep, "__")
                cv2.imwrite(os.path.join(args.viz_dir, f"{tag}__{nid}.jpg"), img)
        total_frames += len(targets)
        print(f"  {root}: {len(originals)} originals copied, {len(targets)} frames added")

    print(f"added {total_frames} frames / {total_boxes} boxes")


if __name__ == "__main__":
    main()