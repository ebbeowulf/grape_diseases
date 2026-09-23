"""Estimate color->thermal extrinsics from clicked correspondences + RealSense depth.

Pipeline
--------
1. For every clicked color point, read the aligned depth image, take the median
   of valid depths in a small window, and reject points whose depth is missing
   or varies too much within the window (typical at leaf/sky edges).
   Back-project to a 3D point in the color camera frame (meters).
2. Load FIXED thermal intrinsics from a previous calibration JSON.
3. Solve the thermal pose with solvePnPRansac, then refine inliers with LM.
   Result: X_thermal = R @ X_color + t, with t in meters.
4. Report reprojection error overall and per image, so images with a shared
   offset (likely sync problems) stand out.
"""
import argparse
import json
import os

import cv2
import numpy as np

# Intel RealSense color intrinsics (1280x720)
default_color_camera_parameters = {
    "w": 1280,
    "h": 720,
    "fl_x": 920.056,
    "fl_y": 920.196,
    "cx": 635.526,
    "cy": 370.034,
    "k1": 0,
    "k2": 0,
    "p1": 0,
    "p2": 0,
}


def load_matches(path):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["color_points"] = np.array(rec["color_points"], dtype=np.float64).reshape(-1, 2)
            rec["thermal_points"] = np.array(rec["thermal_points"], dtype=np.float64).reshape(-1, 2)
            records.append(rec)
    return records


def depth_path_from_color(color_path):
    """.../raw_images/color/color_<ID>.png -> .../raw_images/depth/depth_<ID>.png"""
    color_dir = os.path.dirname(color_path)
    root = os.path.dirname(color_dir)
    image_id = os.path.basename(color_path)[len("color_"):]
    return os.path.join(root, "depth", "depth_" + image_id)


def sample_depth(depth_m, u, v, win, min_valid_frac, max_rel_spread, min_depth, max_depth):
    """Median depth (m) in a win x win box around (u, v), or (None, reason)."""
    h, w = depth_m.shape
    r = win // 2
    x, y = int(round(u)), int(round(v))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, "outside depth image"
    patch = depth_m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].ravel()
    valid = patch[(patch > min_depth) & (patch < max_depth)]
    if len(valid) < min_valid_frac * len(patch):
        return None, "missing depth"
    z = float(np.median(valid))
    spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    if spread > max_rel_spread * z:
        return None, f"depth edge (spread {spread*1000:.0f} mm at {z:.2f} m)"
    return z, None


def sample_box_depth(depth_m, box, inner_frac, min_valid_frac, max_rel_spread,
                     min_depth, max_depth):
    """Median depth (m) over the inner part of a box, or (None, reason).

    Only the central `inner_frac` of the box is used, so background pixels
    around the object's outline stay out of the estimate.
    """
    h, w = depth_m.shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(x1 - x0, 1.0) * inner_frac, max(y1 - y0, 1.0) * inner_frac
    xa, xb = int(round(cx - bw / 2)), int(round(cx + bw / 2)) + 1
    ya, yb = int(round(cy - bh / 2)), int(round(cy + bh / 2)) + 1
    xa, xb = max(0, xa), min(w, xb)
    ya, yb = max(0, ya), min(h, yb)
    if xb <= xa or yb <= ya:
        return None, "box outside depth image"
    patch = depth_m[ya:yb, xa:xb].ravel()
    valid = patch[(patch > min_depth) & (patch < max_depth)]
    if len(valid) < min_valid_frac * len(patch):
        return None, f"missing depth ({len(valid)}/{len(patch)} valid)"
    z = float(np.median(valid))
    spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    if spread > max_rel_spread * z:
        return None, f"mixed depth (spread {spread*1000:.0f} mm at {z:.2f} m)"
    return z, None


def record_entries(rec):
    """Split a record into box entries and plain point entries.

    Box centers are duplicated into the point fields by the annotation tool, so
    points that coincide with a box center are dropped here.
    """
    entries = []
    boxes_c = rec.get("color_boxes", []) or []
    boxes_t = rec.get("thermal_boxes", []) or []
    centers = []
    for bc, bt in zip(boxes_c, boxes_t):
        entries.append({"kind": "box", "color_box": list(bc), "thermal_box": list(bt),
                        "color": [(bc[0] + bc[2]) / 2.0, (bc[1] + bc[3]) / 2.0],
                        "thermal": [(bt[0] + bt[2]) / 2.0, (bt[1] + bt[3]) / 2.0]})
        centers.append(entries[-1]["color"])
    for (u, v), pt_t in zip(rec["color_points"], rec["thermal_points"]):
        if any(abs(u - c[0]) < 1e-6 and abs(v - c[1]) < 1e-6 for c in centers):
            continue
        entries.append({"kind": "point", "color": [float(u), float(v)],
                        "thermal": [float(pt_t[0]), float(pt_t[1])]})
    return entries


def build_correspondences(records, K_c, dist_c, depth_scale=0.001, depth_win=7,
                          min_valid_frac=0.5, max_rel_spread=0.05, min_depth=0.2,
                          max_depth=6.0, inner_frac=0.6, verbose=True):
    """3D color-frame points + thermal pixels for every usable annotation.

    Boxes take their depth from the box interior; points from a small window.
    Returns (obj_pts, img_pts, rec_idx, entries) where entries[i] also carries
    the boxes and depth, for box evaluation.
    """
    obj_pts, img_pts, rec_idx, kept = [], [], [], []
    n_rejected = 0
    K_inv = np.linalg.inv(K_c)
    for k, rec in enumerate(records):
        dpath = depth_path_from_color(rec["color"])
        depth_raw = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            if verbose:
                print(f"  {os.path.basename(rec['color'])}: could not read {dpath}, skipping")
            n_rejected += len(rec["color_points"])
            continue
        depth_m = depth_raw.astype(np.float64) * depth_scale

        for ent in record_entries(rec):
            u, v = ent["color"]
            if ent["kind"] == "box":
                z, reason = sample_box_depth(depth_m, ent["color_box"], inner_frac,
                                             min_valid_frac, max_rel_spread, min_depth, max_depth)
            else:
                z, reason = sample_depth(depth_m, u, v, depth_win, min_valid_frac,
                                         max_rel_spread, min_depth, max_depth)
            if z is None:
                if verbose:
                    print(f"  {os.path.basename(rec['color'])}: reject {ent['kind']} "
                          f"({u:.0f},{v:.0f}) - {reason}")
                n_rejected += 1
                continue
            xn = cv2.undistortPoints(np.array([[[u, v]]]), K_c, dist_c).reshape(2)
            obj_pts.append([xn[0] * z, xn[1] * z, z])
            img_pts.append(ent["thermal"])
            rec_idx.append(k)
            ent = dict(ent, z=z, rec=k)
            kept.append(ent)
    return (np.array(obj_pts, dtype=np.float64).reshape(-1, 3),
            np.array(img_pts, dtype=np.float64).reshape(-1, 2),
            np.array(rec_idx, dtype=int), kept, n_rejected)


def fit_pose(obj_pts, img_pts, K_t, dist_t, th_size, ransac_px=4.0, refine="none", verbose=True):
    """RANSAC PnP with fixed intrinsics, then optional joint intrinsics refinement.

    Returns (R, t, K_t, dist_t, inlier_mask).
    """
    cv2.setRNGSeed(0)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K_t, dist_t, iterationsCount=5000,
        reprojectionError=ransac_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:
        raise RuntimeError("solvePnPRansac failed")
    inl = np.zeros(len(obj_pts), dtype=bool)
    inl[inliers.ravel()] = True
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inl], img_pts[inl], K_t, dist_t, rvec, tvec)

    # A wrong focal length is partly absorbed by moving the thermal camera
    # forward/back (lens z), which only works at one depth. With points at
    # several depths, focal length and lens z can be separated. Inliers are
    # re-selected under each updated model, since the fixed-K inlier set is
    # biased toward the old focal length.
    if refine != "none":
        f_fixed = float(K_t[0, 0])
        if verbose:
            C = -cv2.Rodrigues(rvec)[0].T @ tvec.ravel()
            print(f"\nfixed intrinsics: f {K_t[0, 0]:.1f}  pp ({K_t[0, 2]:.1f},{K_t[1, 2]:.1f})  "
                  f"lens position (mm) {np.array2string(C * 1000, precision=1)}")
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
                 cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        if refine == "focal":
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        dist_guess = np.zeros(5) if dist_t is None else np.resize(dist_t, 5)
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 1000, 1e-12)
        K_ref = K_t.copy()
        prev_inl = None
        for it in range(10):
            _, K_ref, dist_ref, rvecs, tvecs = cv2.calibrateCamera(
                [obj_pts[inl].astype(np.float32)], [img_pts[inl].astype(np.float32)],
                th_size, K_ref.copy(), dist_guess.copy(), flags=flags, criteria=criteria)
            rvec, tvec = rvecs[0], tvecs[0]
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_ref, dist_ref)
            inl = np.linalg.norm(img_pts - proj.reshape(-1, 2), axis=1) < ransac_px
            if verbose:
                print(f"  refine iter {it}: f {K_ref[0, 0]:.1f}  "
                      f"pp ({K_ref[0, 2]:.1f},{K_ref[1, 2]:.1f})  inliers {inl.sum()}/{len(inl)}")
            if inl.sum() < 6:
                raise RuntimeError("Refinement left too few inliers")
            if prev_inl is not None and np.array_equal(inl, prev_inl):
                break
            prev_inl = inl.copy()
        K_t = K_ref
        dist_t = None if dist_t is None else dist_ref.ravel()[:len(dist_t)]
        if verbose:
            print(f"focal length change vs previous calibration: "
                  f"{100.0 * (K_t[0, 0] / f_fixed - 1.0):+.1f}%")
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), K_t, dist_t, inl


def fit_pose_locked(obj_pts, img_pts, K_t, dist_t, R0, lens0, locked=(0, 1),
                    refine="none", inlier_px=4.0, verbose=True):
    """Refit with chosen lens axes held at their measured values.

    Parameters solved: rotation, the unlocked lens axes, and (per `refine`) the
    thermal focal length and principal point. `lens0` is the thermal lens
    position in the color frame (m); entries listed in `locked` never change.
    Levenberg-Marquardt with a numeric Jacobian, with inliers re-selected under
    each updated model. Returns (R, t, K_t, dist_t, inlier_mask).
    """
    free = [a for a in range(3) if a not in locked]
    ratio = K_t[1, 1] / K_t[0, 0]
    with_focal = refine in ("focal", "focal_pp")
    with_pp = refine == "focal_pp"
    lens0 = np.asarray(lens0, dtype=np.float64)

    def unpack(p):
        rvec = p[:3]
        lens = lens0.copy()
        lens[free] = p[3:3 + len(free)]
        K = K_t.copy()
        i = 3 + len(free)
        if with_focal:
            K[0, 0] = p[i]
            K[1, 1] = p[i] * ratio
            i += 1
        if with_pp:
            K[0, 2], K[1, 2] = p[i], p[i + 1]
        return rvec, lens, K

    def project(p, sel):
        rvec, lens, K = unpack(p)
        R = cv2.Rodrigues(rvec)[0]
        proj, _ = cv2.projectPoints(obj_pts[sel], rvec, -R @ lens, K, dist_t)
        return proj.reshape(-1, 2)

    p = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), lens0[free],
                        [K_t[0, 0]] if with_focal else [],
                        [K_t[0, 2], K_t[1, 2]] if with_pp else []])
    eps = np.array([1e-6] * 3 + [1e-6] * len(free) +
                   ([1e-3] if with_focal else []) + ([1e-3, 1e-3] if with_pp else []))

    all_idx = np.ones(len(obj_pts), dtype=bool)
    use = np.linalg.norm(img_pts - project(p, all_idx), axis=1) < inlier_px
    if use.sum() < 6:
        use = all_idx.copy()

    for outer in range(5):
        lam = 1e-3
        r = (project(p, use) - img_pts[use]).ravel()
        cost = r @ r
        for _ in range(200):
            J = np.empty((len(r), len(p)))
            for i in range(len(p)):
                dp = np.zeros(len(p))
                dp[i] = eps[i]
                J[:, i] = ((project(p + dp, use) - img_pts[use]).ravel() - r) / eps[i]
            step = -np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12), J.T @ r)
            r_new = (project(p + step, use) - img_pts[use]).ravel()
            if r_new @ r_new < cost:
                converged = cost - r_new @ r_new < 1e-10 * max(cost, 1.0)
                p, r, cost = p + step, r_new, r_new @ r_new
                lam = max(lam / 10.0, 1e-9)
                if converged:
                    break
            else:
                lam *= 10.0
                if lam > 1e8:
                    break
        new_use = np.linalg.norm(img_pts - project(p, all_idx), axis=1) < inlier_px
        rvec, lens, K = unpack(p)
        if verbose:
            print(f"  locked fit iter {outer}: f {K[0, 0]:.1f}  pp ({K[0, 2]:.1f},{K[1, 2]:.1f})  "
                  f"lens (mm) {np.array2string(lens * 1000, precision=1)}  "
                  f"inliers {new_use.sum()}/{len(new_use)}")
        if new_use.sum() < 6 or np.array_equal(new_use, use):
            break
        use = new_use

    rvec, lens, K = unpack(p)
    R = cv2.Rodrigues(rvec)[0]
    inl = np.linalg.norm(img_pts - project(p, all_idx), axis=1) < inlier_px
    return R, -R @ lens, K, dist_t, inl


def project_box(box, z, K_c, dist_c, K_t, dist_t, R, t, grid=5):
    """Project a color-image box at depth z into the thermal image.

    Samples a grid over the box so lens distortion and perspective are included,
    then takes the bounding box of the projections. Returns None if it falls
    behind the thermal camera.
    """
    x0, y0, x1, y1 = box
    xs = np.linspace(x0, x1, grid)
    ys = np.linspace(y0, y1, grid)
    uv = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    xn = cv2.undistortPoints(uv.reshape(-1, 1, 2), K_c, dist_c).reshape(-1, 2)
    X_c = np.hstack([xn * z, np.full((len(xn), 1), z)])
    X_t = X_c @ R.T + t.reshape(1, 3)
    if (X_t[:, 2] <= 1e-6).any():
        return None
    proj, _ = cv2.projectPoints(X_c, cv2.Rodrigues(R)[0], t, K_t, dist_t)
    p = proj.reshape(-1, 2)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_thermal_intrinsics(path):
    with open(path) as fh:
        calib = json.load(fh)
    K_t = np.array(calib["K_thermal"], dtype=np.float64)
    dist_t = calib.get("dist_thermal")
    dist_t = None if dist_t is None else np.array(dist_t, dtype=np.float64)
    size = None
    if "thermal" in calib and "w" in calib["thermal"]:
        size = (int(calib["thermal"]["w"]), int(calib["thermal"]["h"]))
    return calib, K_t, dist_t, size


def main():
    parser = argparse.ArgumentParser(description="Color->thermal pose from clicked matches + aligned depth")
    parser.add_argument("--matches_dir", type=str, default=".",
                        help="directory containing the shared matches file")
    parser.add_argument("--matches", type=str, default="thermal_color_matches.jsonl",
                        help="matches filename (JSON lines from align_color_thermal.py)")
    parser.add_argument("--calib_dir", type=str, default=".",
                        help="directory containing the previous calibration JSON")
    parser.add_argument("--calib", type=str, default="thermal_color_matches_calib.json",
                        help="previous calibration JSON providing K_thermal (held fixed)")
    parser.add_argument("--out", type=str, default="thermal_color_pose_calib.json",
                        help="output calibration filename, written into matches_dir")
    parser.add_argument("--depth_scale", type=float, default=0.001,
                        help="meters per depth unit (RealSense mm = 0.001)")
    parser.add_argument("--depth_win", type=int, default=7,
                        help="side length (px) of the depth sampling window")
    parser.add_argument("--min_valid_frac", type=float, default=0.5,
                        help="minimum fraction of valid depth pixels in the window")
    parser.add_argument("--inner_frac", type=float, default=0.6,
                        help="fraction of a box's size used for its depth estimate")
    parser.add_argument("--max_rel_spread", type=float, default=0.05,
                        help="reject if (p90-p10) depth spread exceeds this fraction of median depth")
    parser.add_argument("--min_depth", type=float, default=0.2,
                        help="minimum usable depth (m)")
    parser.add_argument("--max_depth", type=float, default=6.0,
                        help="maximum usable depth (m)")
    parser.add_argument("--ransac_px", type=float, default=4.0,
                        help="RANSAC reprojection threshold in thermal pixels")
    parser.add_argument("--offset_flag_px", type=float, default=5.0,
                        help="flag images whose mean thermal residual exceeds this (px)")
    parser.add_argument("--lens_mm", type=float, nargs=3, default=None,
                        help="measured thermal lens position in the color frame, mm: "
                             "x (right+) y (down+) z (forward+); axes in --lock are held there")
    parser.add_argument("--lock", type=str, default="xy",
                        help="lens axes held fixed when --lens_mm is given: any of x, y, z "
                             "(empty string uses --lens_mm only as a starting point)")
    parser.add_argument("--focal_scale", type=float, default=1.0,
                        help="multiply the starting thermal focal length by this before fitting "
                             "(e.g. 1/1.09 if evaluate_box_mapping reports a size ratio of 1.09)")
    parser.add_argument("--refine", type=str, default="none", choices=["none", "focal", "focal_pp"],
                        help="after the fixed-intrinsics fit, also refine thermal focal length "
                             "(focal) or focal length + principal point (focal_pp) jointly with the pose")
    args = parser.parse_args()

    matches_path = args.matches_dir + "/" + args.matches
    calib_path = args.calib_dir + "/" + args.calib
    out_path = args.matches_dir + "/" + args.out

    cp = default_color_camera_parameters
    K_c = np.array([[cp["fl_x"], 0.0, cp["cx"]],
                    [0.0, cp["fl_y"], cp["cy"]],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_c = np.array([cp["k1"], cp["k2"], cp["p1"], cp["p2"]], dtype=np.float64)

    # --- 2. fixed thermal intrinsics --------------------------------------
    prev_calib, K_t, dist_t, prev_size = load_thermal_intrinsics(calib_path)
    if args.focal_scale != 1.0:
        K_t[0, 0] *= args.focal_scale
        K_t[1, 1] *= args.focal_scale
        print(f"focal length scaled by {args.focal_scale:.4f}")
    print("thermal intrinsics (start):\n", np.array2string(K_t, precision=3))

    records = load_matches(matches_path)
    if not records:
        raise RuntimeError(f"No matches found in {matches_path}")

    th_img = cv2.imread(records[0]["thermal"], cv2.IMREAD_UNCHANGED)
    if th_img is None:
        raise FileNotFoundError(f"Could not read thermal image {records[0]['thermal']}")
    th_h, th_w = th_img.shape[:2]
    if prev_size is not None and prev_size != (th_w, th_h):
        print(f"WARNING: previous calibration was {prev_size[0]}x{prev_size[1]}, "
              f"current thermal images are {th_w}x{th_h}; intrinsics may not apply")

    # --- 1. back-project annotations with depth ---------------------------
    obj_pts, img_pts, rec_idx, entries, n_rejected = build_correspondences(
        records, K_c, dist_c, args.depth_scale, args.depth_win, args.min_valid_frac,
        args.max_rel_spread, args.min_depth, args.max_depth, args.inner_frac)
    n_boxes = sum(1 for e in entries if e["kind"] == "box")
    print(f"\n{len(obj_pts)} usable annotations ({n_boxes} boxes, {n_rejected} rejected) "
          f"from {len(records)} image pairs")
    if len(obj_pts) < 6:
        raise RuntimeError("Too few usable annotations for a pose estimate")
    print(f"depth range of used annotations: {obj_pts[:, 2].min():.2f} - {obj_pts[:, 2].max():.2f} m")

    # --- 2b/3. pose fit (+ optional intrinsics refinement) ----------------
    if args.lens_mm is None:
        R, t, K_t, dist_t, inl = fit_pose(obj_pts, img_pts, K_t, dist_t, (th_w, th_h),
                                          args.ransac_px, args.refine)
    else:
        # RANSAC first for outlier rejection and a starting rotation, then refit
        # with the measured lens axes held fixed.
        R, t, K_t, dist_t, inl = fit_pose(obj_pts, img_pts, K_t, dist_t, (th_w, th_h),
                                          args.ransac_px, refine="none")
        locked = tuple("xyz".index(c) for c in args.lock.lower() if c in "xyz")
        lens0 = np.array(args.lens_mm, dtype=np.float64) / 1000.0
        print(f"\nholding lens axes {''.join('xyz'[a] for a in locked) or 'none'} at "
              f"{np.array2string(lens0 * 1000, precision=1)} mm")
        R, t, K_t, dist_t, inl = fit_pose_locked(obj_pts, img_pts, K_t, dist_t, R, lens0,
                                                 locked, args.refine, args.ransac_px)
    rvec, tvec = cv2.Rodrigues(R)[0], t.reshape(3, 1)

    # --- 4. diagnostics ---------------------------------------------------
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_t, dist_t)
    resid = img_pts - proj.reshape(-1, 2)
    err = np.linalg.norm(resid, axis=1)
    thermal_center_in_color = -R.T @ t  # thermal lens position in color frame

    print(f"\nPnP inliers: {inl.sum()}/{len(inl)}  (threshold {args.ransac_px} px)")
    print(f"reprojection error  inliers: median {np.median(err[inl]):.2f}  "
          f"mean {err[inl].mean():.2f} px | all: median {np.median(err):.2f}  "
          f"mean {err.mean():.2f} px")
    print("R (color->thermal):\n", np.array2string(R, precision=5))
    print("rotation vector (deg):", np.array2string(np.degrees(rvec.ravel()), precision=2))
    print("t (m):", np.array2string(t, precision=4))
    print("thermal lens position in color frame (mm, x right / y down / z forward):",
          np.array2string(thermal_center_in_color * 1000, precision=1))

    print("\nper-image residuals (thermal px):")
    per_image = []
    for k, rec in enumerate(records):
        sel = rec_idx == k
        if not sel.any():
            continue
        mean_off = resid[sel].mean(axis=0)
        rms = float(np.sqrt((err[sel] ** 2).mean()))
        flag = "  <-- shared offset" if np.linalg.norm(mean_off) > args.offset_flag_px else ""
        print(f"  {os.path.basename(rec['color']):>18s}  n={int(sel.sum())}  "
              f"inliers={int(inl[sel].sum())}  mean offset ({mean_off[0]:+5.1f},{mean_off[1]:+5.1f})  "
              f"rms {rms:5.1f}{flag}")
        per_image.append({"color": rec["color"], "n": int(sel.sum()),
                          "n_inliers": int(inl[sel].sum()),
                          "mean_offset_px": mean_off.tolist(), "rms_px": rms})

    # Same layout as the existing tool's calibration JSON. The existing overlay
    # reads t_unit as meters, so the metric t is stored there as well.
    result = {
        "thermal": dict(prev_calib.get("thermal", {}), w=int(th_w), h=int(th_h),
                        fl_x=float(K_t[0, 0]), fl_y=float(K_t[1, 1]),
                        cx=float(K_t[0, 2]), cy=float(K_t[1, 2]),
                        model=("pinhole, intrinsics fixed from " if args.refine == "none"
                               else f"pinhole, refine={args.refine} starting from ")
                              + os.path.abspath(calib_path)),
        "extrinsics_color_to_thermal": {
            "R": R.tolist(),
            "t_m": t.tolist(),
            "t_unit": t.tolist(),
            "note": "X_thermal = R @ X_color + t, t in meters (t_unit holds the same metric t)",
        },
        "K_thermal": K_t.tolist(),
        "dist_thermal": None if dist_t is None else dist_t.tolist(),
        "K_color": K_c.tolist(),
        "dist_color": dist_c.tolist(),
        "diagnostics": {
            "n_points_used": int(len(obj_pts)),
            "n_points_rejected": int(n_rejected),
            "n_inliers": int(inl.sum()),
            "ransac_px": args.ransac_px,
            "median_reproj_px_inliers": float(np.median(err[inl])),
            "median_reproj_px_all": float(np.median(err)),
            "thermal_center_in_color_mm": (thermal_center_in_color * 1000).tolist(),
            "per_image": per_image,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()