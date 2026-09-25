#!/usr/bin/env python3
"""Show the boxes leaf_nn_classifier.py trains and evaluates on.

Frames are loaded with the classifier's own dataset constructor and boxes are
chosen by its frame_samples(), so what is drawn is exactly what build_crops
extracts, on the letterboxed frame the crops are cut from:

    red     positive boxes
    cyan    negative boxes kept
    gray    negatives dropped for overlapping a positive
    yellow  negatives removed by the --max-negatives cap

A thin square around each kept box shows the padded, clamped window that is
actually cropped and fed to the network.

Keys: n or space next, p previous, c toggle crop windows, q or Esc quit.
"""

import argparse
import os

import cv2
import numpy as np

from leaf_nn_classifier import (POSITIVE, build_class_map, frame_rng,
                                frame_samples, make_dataset)
from multispectral_detection_loader import IMAGENET_MEAN, IMAGENET_STD

WINDOW = "leaf boxes"
POSITIVE_COLOR = (0, 0, 255)
NEGATIVE_COLOR = (255, 255, 0)
OVERLAP_COLOR = (128, 128, 128)
CAPPED_COLOR = (0, 255, 255)


def to_bgr(pixel_values):
    """Undo the loader's ImageNet normalization; the loader emits RGB."""
    rgb = (pixel_values * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    rgb = (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_box(canvas, box, color, scale, thickness):
    x0, y0, x1, y1 = (int(round(float(v) * scale)) for v in box)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)


def draw_text(canvas, text, y):
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def render(record, name, box_path, index, args, show_windows):
    kept, overlapping, capped = frame_samples(
        record, ["color"], args.size, args.pad, args.max_negatives,
        frame_rng(args.seed, index))

    canvas = to_bgr(record["pixel_values"])
    canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale,
                        interpolation=cv2.INTER_NEAREST)

    for box in overlapping:
        draw_box(canvas, box, OVERLAP_COLOR, args.scale, 1)
    for box in capped:
        draw_box(canvas, box, CAPPED_COLOR, args.scale, 1)
    for box, label, (x, y, side), _ in kept:
        color = POSITIVE_COLOR if label == POSITIVE else NEGATIVE_COLOR
        draw_box(canvas, box, color, args.scale, 2)
        if show_windows:
            draw_box(canvas, (x, y, x + side, y + side), color, args.scale, 1)

    n_pos = sum(1 for k in kept if k[1] == POSITIVE)
    draw_text(canvas, f"[{index}] {name}", 18)
    draw_text(canvas, f"positive {n_pos}  negative {len(kept) - n_pos}  "
                      f"overlap {len(overlapping)}  capped {len(capped)}", 38)
    if not os.path.exists(box_path):
        draw_text(canvas, "NO BOX FILE (see terminal)", 58)
        folder = os.path.dirname(box_path)
        print(f"[{index}] no box file: {box_path}")
        print(f"      folder {'exists' if os.path.isdir(folder) else 'MISSING'}: "
              f"{folder}")
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=str, required=True,
                   help="dataset list file, '<root> <img_id>' per line")
    p.add_argument("--calibration", type=str, required=True,
                   help="camera calibration json for the thermal warp")
    p.add_argument("--box-dir", type=str, default="combined_roboflow_sam3",
                   help="subdirectory under each root holding box pickles")
    p.add_argument("--positive-classes", type=str, nargs="+", required=True,
                   help="box class names forming the positive class")
    p.add_argument("--negative-classes", type=str, nargs="+", required=True,
                   help="box class names forming the negative class")
    p.add_argument("--max-negatives", type=int, default=8,
                   help="cap on negative boxes kept per frame; 0 keeps all")
    p.add_argument("--size", type=int, default=224,
                   help="crop resolution, as in the classifier")
    p.add_argument("--pad", type=float, default=0.1,
                   help="context margin around each box, as in the classifier")
    p.add_argument("--frame-size", type=int, default=576,
                   help="letterboxed frame size from the loader")
    p.add_argument("--seed", type=int, default=0,
                   help="negative-sampling seed; the classifier uses seed "
                        "for train and seed + 1 for eval")
    p.add_argument("--start", type=int, default=0,
                   help="index in the list file of the first frame shown")
    p.add_argument("--image-id", type=str, default="",
                   help="start at the first list entry with this img_id "
                        "(e.g. 00035); overrides --start")
    p.add_argument("--scale", type=float, default=1.5,
                   help="display magnification of the letterboxed frame")
    args = p.parse_args()

    class_map = build_class_map(args.positive_classes, args.negative_classes)
    dataset = make_dataset(args.samples, args.calibration, class_map,
                           args.frame_size, False, args.box_dir)

    index = args.start
    if args.image_id:
        matches = [i for i, (_, img_id) in enumerate(dataset.samples)
                   if img_id == args.image_id]
        if not matches:
            raise SystemExit(f"img_id {args.image_id} not in {args.samples}")
        index = matches[0]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    step, show_windows = 1, True
    loaded, record = None, None
    while 0 <= index < len(dataset):
        if loaded != index:
            record, loaded = dataset[index], index
        if record is None:
            print(f"[{index}] frame skipped by the loader")
            index += step
            if index < 0:
                index, step = 0, 1
            continue

        root, img_id = dataset.samples[index]
        box_path = dataset._paths(root, img_id)[2]
        cv2.imshow(WINDOW, render(record, f"{root} {img_id}", box_path,
                                  index, args, show_windows))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            show_windows = not show_windows
        elif key == ord("p"):
            step, index = -1, max(0, index - 1)
        else:
            step, index = 1, index + 1

    if dataset.missing_boxes:
        print(f"{len(dataset.missing_boxes)} frames viewed had no box "
              f"pickle, e.g. {sorted(dataset.missing_boxes)[0]}")
    if dataset.unmapped:
        print(f"names seen but not in class_map: {sorted(dataset.unmapped)}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()