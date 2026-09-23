"""Show projected color boxes on the thermal image, next to the color image.

Everything is drawn in the THERMAL frame at its native resolution, just zoomed:
nothing is warped, nothing is cropped by depth coverage. This is what the box
mapping actually produces.

    green : the thermal box you annotated
    red   : the color box projected with the calibration, at the box's depth
    cyan  : the same projection after removing this image's median offset
            (toggle with 'o') -- what a per-frame shift correction would give

Keys
    n / b : next / previous image
    c     : show/hide the color panel
    o     : toggle the per-image shift correction
    i     : toggle box labels (index and IoU)
    +/-   : thermal zoom
    w     : write the current view to a PNG
    q/esc : quit
"""
import argparse
import json
import os

import cv2
import numpy as np

import estimate_mapping_from_boxes as est
import view_thermal_color_depth_overlay as viewer


def draw_boxes(canvas, boxes, color, zoom, labels=None, thickness=2):
    for i, b in enumerate(boxes):
        p0 = (int(round(b[0] * zoom)), int(round(b[1] * zoom)))
        p1 = (int(round(b[2] * zoom)), int(round(b[3] * zoom)))
        cv2.rectangle(canvas, p0, p1, color, thickness)
        if labels is not None:
            cv2.putText(canvas, labels[i], (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="View projected boxes in the thermal frame")
    parser.add_argument("--calib_dir", type=str, default=".",
                        help="directory containing the calibration JSON")
    parser.add_argument("--calib", type=str, default="thermal_color_pose_calib.json",
                        help="calibration JSON to project with")
    parser.add_argument("--matches_dir", type=str, default=".",
                        help="directory containing the annotation file; PNGs are written here")
    parser.add_argument("--matches", type=str, default="thermal_color_boxes.jsonl",
                        help="annotation filename with boxes")
    parser.add_argument("--zoom", type=int, default=3,
                        help="thermal display zoom factor")
    parser.add_argument("--depth_scale", type=float, default=0.001,
                        help="meters per depth unit (RealSense mm = 0.001)")
    parser.add_argument("--depth_win", type=int, default=7,
                        help="side length (px) of the depth window for plain points")
    parser.add_argument("--inner_frac", type=float, default=0.6,
                        help="fraction of a box's size used for its depth estimate")
    parser.add_argument("--min_valid_frac", type=float, default=0.5,
                        help="minimum fraction of valid depth pixels")
    parser.add_argument("--max_rel_spread", type=float, default=0.05,
                        help="reject if depth spread exceeds this fraction of median depth")
    parser.add_argument("--min_depth", type=float, default=0.2,
                        help="minimum usable depth (m)")
    parser.add_argument("--max_depth", type=float, default=6.0,
                        help="maximum usable depth (m)")
    args = parser.parse_args()

    calib, K_t, dist_t, _ = est.load_thermal_intrinsics(args.calib_dir + "/" + args.calib)
    K_c = np.array(calib["K_color"], dtype=np.float64)
    dist_c = np.array(calib.get("dist_color", [0, 0, 0, 0]), dtype=np.float64)
    ext = calib["extrinsics_color_to_thermal"]
    R = np.array(ext["R"], dtype=np.float64)
    t = np.array(ext.get("t_m", ext["t_unit"]), dtype=np.float64)

    records = est.load_matches(args.matches_dir + "/" + args.matches)
    _, _, _, entries, _ = est.build_correspondences(
        records, K_c, dist_c, args.depth_scale, args.depth_win, args.min_valid_frac,
        args.max_rel_spread, args.min_depth, args.max_depth, args.inner_frac, verbose=False)

    # project every box once, grouped by image
    per_image = {}
    for ent in entries:
        if ent["kind"] != "box":
            continue
        proj = est.project_box(ent["color_box"], ent["z"], K_c, dist_c, K_t, dist_t, R, t)
        if proj is None:
            continue
        gt = np.array(ent["thermal_box"], dtype=np.float64)
        pc = np.array([(proj[0] + proj[2]) / 2, (proj[1] + proj[3]) / 2])
        gc = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2])
        per_image.setdefault(ent["rec"], []).append(
            {"color_box": np.array(ent["color_box"]), "gt": gt, "proj": proj,
             "offset": pc - gc, "z": ent["z"], "iou": est.box_iou(proj, gt)})
    keys = [k for k in sorted(per_image) if per_image[k]]
    if not keys:
        raise RuntimeError("No boxes could be projected")
    print(f"{sum(len(v) for v in per_image.values())} boxes across {len(keys)} images")

    ui = {"idx": 0, "color": True, "shift": False, "labels": True, "zoom": args.zoom}
    win = "box overlay  [n/b:image  c:color  o:shift  i:labels  +/-:zoom  w:write  q:quit]"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        k = keys[ui["idx"]]
        rec = records[k]
        boxes = per_image[k]
        thermal_raw = cv2.imread(rec["thermal"], cv2.IMREAD_UNCHANGED)
        color = cv2.imread(rec["color"])
        if thermal_raw is None or color is None:
            print(f"could not read {rec['thermal']}")
            break
        th8 = viewer.prep_thermal_for_display(thermal_raw)
        zoom = ui["zoom"]
        panel_t = cv2.resize(cv2.cvtColor(th8, cv2.COLOR_GRAY2BGR), None, fx=zoom, fy=zoom,
                             interpolation=cv2.INTER_NEAREST)

        med = np.median(np.array([b["offset"] for b in boxes]), axis=0)
        shift = np.array([med[0], med[1], med[0], med[1]]) if ui["shift"] else np.zeros(4)
        gt_boxes = [b["gt"] for b in boxes]
        pr_boxes = [b["proj"] - shift for b in boxes]
        ious = [est.box_iou(p, g) for p, g in zip(pr_boxes, gt_boxes)]
        labels = [f"{i}:{iou:.2f}" for i, iou in enumerate(ious)] if ui["labels"] else None
        draw_boxes(panel_t, gt_boxes, (0, 255, 0), zoom)
        draw_boxes(panel_t, pr_boxes, (255, 255, 0) if ui["shift"] else (0, 0, 255), zoom, labels)

        if ui["color"]:
            h_t = panel_t.shape[0]
            s = h_t / color.shape[0]
            panel_c = cv2.resize(color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            draw_boxes(panel_c, [b["color_box"] for b in boxes], (0, 255, 0), s,
                       labels, thickness=1)
            disp = np.hstack([panel_c, np.zeros((h_t, 20, 3), np.uint8), panel_t])
        else:
            disp = panel_t

        status = [
            f"[{ui['idx'] + 1}/{len(keys)}] {os.path.basename(rec['color'])}   "
            f"boxes {len(boxes)}   depth median {np.median([b['z'] for b in boxes]):.2f} m",
            f"IoU median {np.median(ious):.2f}   image median offset "
            f"({med[0]:+.1f},{med[1]:+.1f}) px   "
            f"shift correction: {'on' if ui['shift'] else 'off'}",
        ]
        cv2.rectangle(disp, (0, 0), (max(cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                                         for s_ in status) + 20, 56), (0, 0, 0), -1)
        for i, s_ in enumerate(status):
            cv2.putText(disp, s_, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" ")):
            ui["idx"] = (ui["idx"] + 1) % len(keys)
        elif key == ord("b"):
            ui["idx"] = (ui["idx"] - 1) % len(keys)
        elif key == ord("c"):
            ui["color"] = not ui["color"]
        elif key == ord("o"):
            ui["shift"] = not ui["shift"]
        elif key == ord("i"):
            ui["labels"] = not ui["labels"]
        elif key in (ord("+"), ord("=")):
            ui["zoom"] = min(8, ui["zoom"] + 1)
        elif key == ord("-"):
            ui["zoom"] = max(1, ui["zoom"] - 1)
        elif key == ord("w"):
            name = os.path.splitext(os.path.basename(rec["color"]))[0]
            out = args.matches_dir + "/" + f"box_overlay_{name}.png"
            cv2.imwrite(out, disp)
            print(f"wrote {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
