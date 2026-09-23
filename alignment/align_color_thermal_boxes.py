"""Annotate matching boxes (or points) in synchronized color / thermal image pairs.

Draw a box around a leaf in the color panel, then the same leaf in the thermal
panel, and repeat. Results append to a shared JSON Lines file, one record per
image pair. Each record carries the boxes AND their centers as points, so
estimate_thermal_pose_depth.py can read the same file unchanged.

A short drag (under --min_drag display px) is recorded as a point instead of a
box, so the same tool works for stationary warm-object calibration shots.

A separate zoom window follows the cursor, since a 10-30 px leaf in a 320x240
thermal image is only ~20-60 px on screen. It covers nothing in the annotation
window and can be closed with z or --zoom 0.

Controls
    drag        : draw the next box (alternates color -> thermal)
    click       : place a point instead of a box (short drag)
    t           : cycle thermal contrast mode (local / pct / clahe16 / std)
    [ / ]       : clip more / less of the thermal range at each end
    c           : cycle the CLAHE clip limit (0 = off)
    e           : toggle unsharp sharpening
    g           : cycle the local-contrast blur radius
    u           : undo the last box/point
    r           : clear this image pair
    z           : show/hide the zoom window
    +/-         : how much area the zoom window covers
    enter / q   : save this pair and go to the next
    esc         : discard this pair and go to the next
    ctrl-c      : stop (already-saved pairs are kept)
"""
import argparse
import json
import os
import re
from datetime import datetime

import cv2
import numpy as np

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
          (255, 0, 255), (255, 255, 0), (255, 255, 255), (0, 128, 255)]


def get_random_image_pairs(root_dir, num_pairs):
    """Return up to num_pairs (color_path, thermal_path) tuples with matching IDs."""
    color_dir = root_dir + "/raw_images/color"
    thermal_dir = root_dir + "/raw_images/thermal"
    if not os.path.isdir(color_dir) or not os.path.isdir(thermal_dir):
        return []
    color_re = re.compile(r"^color_(.+)\.png$", re.IGNORECASE)
    thermal_re = re.compile(r"^thermal_(.+)\.png$", re.IGNORECASE)
    id_from_color, id_from_thermal = {}, {}
    for f in os.listdir(color_dir):
        m = color_re.match(f)
        if m:
            id_from_color[m.group(1)] = os.path.join(color_dir, f)
    for f in os.listdir(thermal_dir):
        m = thermal_re.match(f)
        if m:
            id_from_thermal[m.group(1)] = os.path.join(thermal_dir, f)
    common = sorted(set(id_from_color) & set(id_from_thermal))
    if not common:
        return []
    if num_pairs < len(common):
        common = list(np.random.choice(common, size=num_pairs, replace=False))
    return [(id_from_color[i], id_from_thermal[i]) for i in common]


TONE_MODES = ["local", "pct", "clahe16", "std"]


def prep_thermal_for_display(thermal, mode="local", pct=1.0, n_std=1.5, clip=3.0,
                             tile=8, unsharp=0.0, sigma=15.0):
    """16-bit thermal -> 8-bit gray, with several contrast options.

    mode:
      local   - subtract a heavily blurred copy (sigma px) before stretching, so
                slow temperature gradients stop eating the dynamic range. Best
                on flat, hazy frames where the whole canopy sits in a few counts.
      pct     - stretch between the pct and 100-pct percentiles, then CLAHE.
                Robust to a cold sky or a hot spot pinning the range.
      clahe16 - CLAHE on the raw 16-bit data first, then a percentile stretch,
                so local contrast is computed before any quantization.
      std     - the original mean +/- n_std window, then CLAHE.

    unsharp adds that fraction of a high-pass copy back in, which sharpens
    edges (it cannot recover true optical blur, only make edges easier to see).
    """
    if thermal.ndim == 3:
        thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)

    def stretch(f, lo, hi):
        if hi - lo < 1e-6:
            hi = lo + 1.0
        return (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=max(clip, 0.01), tileGridSize=(tile, tile))
    f = thermal.astype(np.float32)

    if mode == "local":
        f = f - cv2.GaussianBlur(f, (0, 0), sigma)
        out = stretch(f, *np.percentile(f, [pct, 100 - pct]))
        if clip > 0:
            out = clahe.apply(out)
    elif mode == "clahe16":
        raw16 = thermal.astype(np.uint16) if thermal.dtype != np.uint16 else thermal
        eq = clahe.apply(raw16).astype(np.float32)
        out = stretch(eq, *np.percentile(eq, [pct, 100 - pct]))
    elif mode == "pct":
        out = stretch(f, *np.percentile(f, [pct, 100 - pct]))
        if clip > 0:
            out = clahe.apply(out)
    else:  # std
        mean, std = float(f.mean()), float(f.std())
        out = stretch(f, mean - n_std * max(std, 1e-6), mean + n_std * max(std, 1e-6))
        if clip > 0:
            out = clahe.apply(out)

    if unsharp > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = np.clip(out.astype(np.float32) + unsharp * (out.astype(np.float32) - blur),
                      0, 255).astype(np.uint8)
    return out


class Annotator:
    """Side-by-side annotation state. All stored coordinates are ORIGINAL image px."""

    GAP = 20

    def __init__(self, color, thermal_8u, max_width=1500, max_height=850, min_drag=3):
        self.color = color
        self.thermal = cv2.cvtColor(thermal_8u, cv2.COLOR_GRAY2BGR)
        self.min_drag = min_drag
        ch, cw = color.shape[:2]
        th, tw = thermal_8u.shape[:2]

        aspect_sum = (cw / ch) + (tw / th)
        disp_h = int(round(min(max_height, (max_width - self.GAP) / aspect_sum)))
        self.scale_c = disp_h / ch
        self.scale_t = disp_h / th
        cw_d = int(round(cw * self.scale_c))
        tw_d = int(round(tw * self.scale_t))
        self.cw_d = cw_d
        self.thermal_x0 = cw_d + self.GAP
        self.disp_h = disp_h

        self.tw_d = tw_d
        self.base = np.zeros((disp_h, cw_d + self.GAP + tw_d, 3), dtype=np.uint8)
        self.base[:, :cw_d] = cv2.resize(self.color, (cw_d, disp_h), interpolation=cv2.INTER_AREA)
        self.set_thermal(thermal_8u)
        self.pairs = []          # [{"type", "color", "thermal"}], thermal may be missing
        self.expect = "color"    # which panel the next annotation belongs to
        self.drag_start = None   # canvas px
        self.cursor = None       # canvas px
        self.inset_src = 40      # source px shown in the inset
        self.message = ""

    def set_thermal(self, thermal_8u):
        """Swap in a re-rendered thermal image; annotations are unaffected."""
        self.thermal = cv2.cvtColor(thermal_8u, cv2.COLOR_GRAY2BGR)
        self.base[:, self.thermal_x0:] = cv2.resize(self.thermal, (self.tw_d, self.disp_h),
                                                    interpolation=cv2.INTER_NEAREST)

    # --- coordinate helpers -------------------------------------------------
    def panel(self, x):
        if x < self.cw_d:
            return "color"
        if x >= self.thermal_x0:
            return "thermal"
        return None

    def to_image(self, panel, x, y):
        """Canvas px -> original image px, undoing display scaling (pixel centers)."""
        if panel == "color":
            s, x0 = self.scale_c, 0
        else:
            s, x0 = self.scale_t, self.thermal_x0
        return ((x - x0 + 0.5) / s - 0.5, (y + 0.5) / s - 0.5)

    def to_canvas(self, panel, x, y):
        if panel == "color":
            s, x0 = self.scale_c, 0
        else:
            s, x0 = self.scale_t, self.thermal_x0
        return (int(round((x + 0.5) * s - 0.5)) + x0, int(round((y + 0.5) * s - 0.5)))

    # --- mouse ---------------------------------------------------------------
    def on_mouse(self, event, x, y, flags=0, param=None):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.panel(x) == self.expect:
                self.drag_start = (x, y)
            else:
                self.message = f"click inside the {self.expect} panel"
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            self.drag_start = None
            if self.panel(x) != self.expect:
                self.message = f"finish the drag inside the {self.expect} panel"
                return
            drag = max(abs(x - x0), abs(y - y0))
            self.add(self.expect, (x0, y0), (x, y), is_box=drag >= self.min_drag)

    def add(self, panel, p0, p1, is_box):
        a = self.to_image(panel, *p0)
        b = self.to_image(panel, *p1)
        if is_box:
            shape = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
            kind = "box"
        else:
            shape = list(b)
            kind = "point"

        if panel == "color":
            self.pairs.append({"type": kind, "color": shape})
            self.expect = "thermal"
            self.message = ""
        else:
            want = self.pairs[-1]["type"]
            if kind != want:
                self.message = f"the color annotation was a {want}; draw a {want} here"
                return
            self.pairs[-1]["thermal"] = shape
            self.expect = "color"
            self.message = ""

    def undo(self):
        if not self.pairs:
            return
        if "thermal" in self.pairs[-1]:
            del self.pairs[-1]["thermal"]
            self.expect = "thermal"
        else:
            self.pairs.pop()
            self.expect = "color"

    def reset(self):
        self.pairs.clear()
        self.expect = "color"

    # --- drawing -------------------------------------------------------------
    def inset(self, side=300):
        """Magnified view around the cursor, for its own window. None if unavailable."""
        if self.cursor is None:
            return None
        panel = self.panel(self.cursor[0])
        if panel is None:
            return None
        img = self.color if panel == "color" else self.thermal
        cx, cy = self.to_image(panel, *self.cursor)
        h, w = img.shape[:2]
        half = self.inset_src // 2
        x0 = int(np.clip(round(cx) - half, 0, max(0, w - self.inset_src)))
        y0 = int(np.clip(round(cy) - half, 0, max(0, h - self.inset_src)))
        patch = img[y0:y0 + self.inset_src, x0:x0 + self.inset_src]
        if patch.size == 0:
            return None
        inset = cv2.resize(patch, (side, side), interpolation=cv2.INTER_NEAREST)
        s = side / patch.shape[1]

        # annotations inside the inset
        for i, pr in enumerate(self.pairs):
            shape = pr.get(panel)
            if shape is None:
                continue
            c = COLORS[i % len(COLORS)]
            if pr["type"] == "box":
                p = [(shape[0] - x0) * s, (shape[1] - y0) * s,
                     (shape[2] - x0) * s, (shape[3] - y0) * s]
                cv2.rectangle(inset, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), c, 1)
            else:
                cv2.drawMarker(inset, (int((shape[0] - x0) * s), int((shape[1] - y0) * s)),
                               c, cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(inset, (int((cx - x0) * s), int((cy - y0) * s)),
                       (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
        label = f"{panel}  {self.inset_src} px across"
        cv2.rectangle(inset, (0, 0), (side - 1, 20), (0, 0, 0), -1)
        cv2.putText(inset, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return inset

    def render(self):
        disp = self.base.copy()
        for i, pr in enumerate(self.pairs):
            c = COLORS[i % len(COLORS)]
            for panel in ("color", "thermal"):
                shape = pr.get(panel)
                if shape is None:
                    continue
                if pr["type"] == "box":
                    p0 = self.to_canvas(panel, shape[0], shape[1])
                    p1 = self.to_canvas(panel, shape[2], shape[3])
                    cv2.rectangle(disp, p0, p1, c, 2)
                    cv2.putText(disp, str(i), (p0[0] + 3, p0[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
                else:
                    p = self.to_canvas(panel, shape[0], shape[1])
                    cv2.drawMarker(disp, p, c, cv2.MARKER_CROSS, 14, 2)
                    cv2.putText(disp, str(i), (p[0] + 6, p[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if self.drag_start is not None and self.cursor is not None:
            cv2.rectangle(disp, self.drag_start, self.cursor, (255, 255, 255), 1)

        done = sum(1 for p in self.pairs if "thermal" in p)
        status = f"matched: {done}   next: {self.expect}   {self.message}"
        cv2.rectangle(disp, (0, 0), (760, 34), (0, 0, 0), -1)
        cv2.putText(disp, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return disp

    # --- output --------------------------------------------------------------
    def to_record(self, image_pair):
        """Complete pairs as a record: boxes, plus centers in the point fields."""
        boxes_c, boxes_t, pts_c, pts_t = [], [], [], []
        for pr in self.pairs:
            if "thermal" not in pr:
                continue
            if pr["type"] == "box":
                bc, bt = pr["color"], pr["thermal"]
                boxes_c.append(bc)
                boxes_t.append(bt)
                pts_c.append([(bc[0] + bc[2]) / 2.0, (bc[1] + bc[3]) / 2.0])
                pts_t.append([(bt[0] + bt[2]) / 2.0, (bt[1] + bt[3]) / 2.0])
            else:
                pts_c.append(list(pr["color"]))
                pts_t.append(list(pr["thermal"]))
        if not pts_c:
            return None
        return {
            "color": os.path.abspath(image_pair[0]),
            "thermal": os.path.abspath(image_pair[1]),
            "n": len(pts_c),
            "color_points": pts_c,
            "thermal_points": pts_t,
            "n_boxes": len(boxes_c),
            "color_boxes": boxes_c,
            "thermal_boxes": boxes_t,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


def main():
    parser = argparse.ArgumentParser(description="Annotate matching color/thermal boxes")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="bag directory containing raw_images/{color,thermal}")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="directory the annotation file is written to")
    parser.add_argument("--out", type=str, default="thermal_color_boxes.jsonl",
                        help="annotation filename; records are appended, never overwritten")
    parser.add_argument("--num_pairs", type=int, default=10,
                        help="number of random image pairs to annotate")
    parser.add_argument("--max_width", type=int, default=1500,
                        help="maximum canvas width in screen px")
    parser.add_argument("--max_height", type=int, default=850,
                        help="maximum canvas height in screen px")
    parser.add_argument("--zoom", type=int, default=1,
                        help="1 to open the magnified zoom window at startup, 0 to start without it")
    parser.add_argument("--zoom_size", type=int, default=300,
                        help="side length in px of the zoom window")
    parser.add_argument("--min_drag", type=int, default=3,
                        help="drags shorter than this many display px are saved as points")
    parser.add_argument("--tone", type=str, default="local", choices=TONE_MODES,
                        help="thermal contrast mode at startup (cycle with t)")
    parser.add_argument("--pct", type=float, default=1.0,
                        help="percentile clipped at each end of the thermal range")
    parser.add_argument("--clip", type=float, default=3.0,
                        help="CLAHE clip limit; 0 disables CLAHE")
    parser.add_argument("--tile", type=int, default=8,
                        help="CLAHE tile grid size")
    parser.add_argument("--unsharp", type=float, default=0.0,
                        help="unsharp amount added to the thermal display (0 = off)")
    parser.add_argument("--sigma", type=float, default=15.0,
                        help="blur radius (px) removed in the local contrast mode")
    parser.add_argument("--n_std", type=float, default=1.5,
                        help="thermal display window as mean +/- n_std * std (std mode)")
    args = parser.parse_args()

    out_path = args.out_dir + "/" + args.out
    pairs = get_random_image_pairs(args.root_dir, args.num_pairs)
    if not pairs:
        raise RuntimeError(f"No matching color/thermal pairs under {args.root_dir}")

    win = "annotate boxes  [drag:box  click:point  u:undo  r:reset  z:zoom  enter:save  esc:skip]"
    zoom_win = "zoom"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    zoom = {"on": args.zoom != 0}
    if zoom["on"]:
        cv2.namedWindow(zoom_win, cv2.WINDOW_AUTOSIZE)

    for n, pair in enumerate(pairs, 1):
        color = cv2.imread(pair[0])
        thermal_raw = cv2.imread(pair[1], cv2.IMREAD_UNCHANGED)
        if color is None or thermal_raw is None:
            print(f"skipping {pair[0]}: could not read color/thermal")
            continue
        tone = {"mode": args.tone, "pct": args.pct, "clip": args.clip,
                "tile": args.tile, "unsharp": args.unsharp, "sigma": args.sigma}

        def render_thermal():
            return prep_thermal_for_display(thermal_raw, tone["mode"], tone["pct"], args.n_std,
                                            tone["clip"], tone["tile"], tone["unsharp"],
                                            tone["sigma"])

        ann = Annotator(color, render_thermal(), args.max_width, args.max_height, args.min_drag)
        cv2.setMouseCallback(win, ann.on_mouse)
        print(f"[{n}/{len(pairs)}] {os.path.basename(pair[0])}")

        save = True
        while True:
            cv2.imshow(win, ann.render())
            if zoom["on"]:
                inset = ann.inset(args.zoom_size)
                if inset is not None:
                    cv2.imshow(zoom_win, inset)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("z"):
                zoom["on"] = not zoom["on"]
                if zoom["on"]:
                    cv2.namedWindow(zoom_win, cv2.WINDOW_AUTOSIZE)
                else:
                    cv2.destroyWindow(zoom_win)
                continue
            if key in (13, ord("q")):
                break
            if key == 27:
                save = False
                break
            if key in (ord("t"), ord("["), ord("]"), ord("c"), ord("e"), ord("g")):
                if key == ord("t"):
                    tone["mode"] = TONE_MODES[(TONE_MODES.index(tone["mode"]) + 1) % len(TONE_MODES)]
                elif key == ord("["):
                    tone["pct"] = min(20.0, tone["pct"] * 2 if tone["pct"] else 0.25)
                elif key == ord("]"):
                    tone["pct"] = max(0.0, tone["pct"] / 2)
                elif key == ord("c"):
                    tone["clip"] = {0.0: 1.0, 1.0: 2.0, 2.0: 3.0, 3.0: 5.0,
                                    5.0: 8.0, 8.0: 0.0}.get(tone["clip"], 3.0)
                elif key == ord("e"):
                    tone["unsharp"] = 0.0 if tone["unsharp"] else 1.0
                else:
                    tone["sigma"] = {5.0: 15.0, 15.0: 30.0, 30.0: 60.0, 60.0: 5.0}.get(
                        tone["sigma"], 15.0)
                ann.set_thermal(render_thermal())
                ann.message = (f"tone {tone['mode']} pct {tone['pct']:g} clip {tone['clip']:g} "
                               f"sigma {tone['sigma']:g}"
                               + (" unsharp" if tone["unsharp"] else ""))
                continue
            if key == ord("u"):
                ann.undo()
            elif key == ord("r"):
                ann.reset()
            elif key in (ord("+"), ord("=")):
                ann.inset_src = max(10, ann.inset_src - 10)
            elif key == ord("-"):
                ann.inset_src = min(120, ann.inset_src + 10)

        record = ann.to_record(pair) if save else None
        if record is None:
            print("  nothing saved for this pair")
            continue
        os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  saved {record['n_boxes']} boxes and {record['n'] - record['n_boxes']} points "
              f"-> {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()