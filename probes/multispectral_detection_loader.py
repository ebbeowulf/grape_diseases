#!/usr/bin/env python3
"""Multi-spectral detection dataset: colour + thermal, one class, no inheritance.

Pipeline per frame:
    load colour and raw thermal
    warp thermal into the colour frame using the calibration
    crop both to the rectangle where thermal data exists
    z-score thermal against vegetation pixels, not the whole frame
    load boxes, map class names to ids, drop unmapped
    letterbox to a fixed square and emit HF detection targets

Classes are mapped by NAME, which is how the box pickles are keyed. Any name
absent from class_map is dropped, and the names encountered but not mapped are
reported by print_category_counts() so a typo shows up instead of silently
removing data.

Depth and segmentation masks are deliberately absent: the fixed-depth warp
scored better than per-pixel depth, and masks play no part in detection.
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class MultiSpectralDetection(Dataset):
    def __init__(self, samples_file, calibration_file, class_map,
                 size=576, depth_baselines=7.0, min_side=2, augment=False,
                 min_green_pixels=500, box_dir="roboflow_boxes",
                 box_prefix="boxes_", use_depth=False,
                 color_dir="color", thermal_dir="thermal", depth_dir="depth",
                 ext=".png"):
        """
        samples_file      list file, one '<root> <img_id>' per line
        calibration_file  json holding both intrinsics and the extrinsics
        class_map         {class name: output id}; -1 drops deliberately,
                          a name absent from the map drops and is reported
        size              square resolution after letterboxing
        depth_baselines   constant warp depth, in calibration baselines --
                          7.0 came from the mutual-information sweep
        min_side          boxes thinner than this after scaling are dropped;
                          slivers become degenerate targets and NaN the loss
        min_green_pixels  below this the thermal z-score has no vegetation
                          to reference and the frame is skipped
        use_depth         also emit depth_values; depth is assumed already
                          registered to the colour frame by the sensor, so
                          it is cropped alongside colour rather than warped
        """
        self.samples = self._read_samples(samples_file)
        self.use_depth = use_depth
        self.depth_dir = depth_dir
        self.class_map = class_map
        self.size = size
        self.depth_baselines = depth_baselines
        self.min_side = min_side
        self.augment = augment
        self.min_green_pixels = min_green_pixels
        self.box_dir, self.box_prefix = box_dir, box_prefix
        self.color_dir, self.thermal_dir, self.ext = color_dir, thermal_dir, ext
        self.unmapped = set()

        with open(calibration_file) as fh:
            calib = json.load(fh)
        self.K_color = np.array(calib["K_color"], dtype=np.float32)
        self.K_thermal = np.array(calib["K_thermal"], dtype=np.float32)
        ext_calib = calib["extrinsics_color_to_thermal"]
        self.R = np.array(ext_calib["R"], dtype=np.float32)
        # t is a unit vector, so depth entering the warp is in baselines.
        self.t = np.array(ext_calib["t_unit"], dtype=np.float32).reshape(3, 1)

    @staticmethod
    def _read_samples(path):
        out = []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    out.append((parts[0], parts[1]))
        if not out:
            raise SystemExit(f"no samples parsed from {path}")
        return out

    def __len__(self):
        return len(self.samples)

    # ---- loading -------------------------------------------------------

    def _paths(self, root, img_id):
        return (
            os.path.join(root, self.color_dir, f"color_{img_id}{self.ext}"),
            os.path.join(root, self.thermal_dir, f"thermal_{img_id}{self.ext}"),
            os.path.join(root, self.box_dir, f"{self.box_prefix}{img_id}.pkl"),
            os.path.join(root, self.depth_dir, f"depth_{img_id}{self.ext}"),
        )

    def _depth_zscore(self, depth, color_bgr, valid):
        """Standardize depth against vegetation, like thermal.

        Holes come back as 0 from the sensor and would drag the statistics
        down, so they are excluded and then filled with the vegetation
        median. What survives is relief relative to the canopy, not absolute
        standoff, which also removes drift between capture sessions.
        """
        b, g, r = cv2.split(color_bgr.astype(np.float32))
        green = (g > b) & (g > r) & valid & (depth > 0)
        if green.sum() < self.min_green_pixels:
            green = valid & (depth > 0)
        if green.sum() < self.min_green_pixels:
            return None

        values = depth[green].astype(np.float32)
        std = values.std()
        if std < 1e-6:
            return None

        out = depth.astype(np.float32)
        out[depth <= 0] = np.median(values)     # fill holes before scaling
        return (out - values.mean()) / std

    def _load_boxes(self, path):
        """One pickle per frame, keyed by class name, values a list of
        (score, [x0, y0, x1, y1]) tuples:

            {"esca": [(1.0, [382.0, 227.3, 404.0, 252.2]), ...],
             "grass": []}

        The score comes from the SAM3 pipeline and is 1.0 for converted
        Roboflow annotations, so it is read and discarded. This is the only
        place the annotation format is assumed."""
        if not os.path.exists(path):
            return np.zeros((0, 4), np.float32), []
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        boxes, names = [], []
        for name, entries in data.items():
            for entry in entries:
                box = entry[1] if isinstance(entry, (tuple, list)) and len(entry) == 2 \
                    else entry
                boxes.append(np.asarray(box, dtype=np.float32).reshape(4))
                names.append(name)
        return np.array(boxes, np.float32).reshape(-1, 4), names

    # ---- geometry ------------------------------------------------------

    def _warp_thermal(self, thermal, color_shape):
        """Project thermal into the colour frame at a constant depth.

        For every colour pixel: back-project to a 3D ray, push it out to the
        assumed depth, move it into the thermal camera's frame, and project.
        The result is a lookup table saying where each colour pixel lands in
        the thermal image, which cv2.remap then samples.
        """
        h_c, w_c = color_shape
        h_t, w_t = thermal.shape[:2]

        # Homogeneous pixel coordinates for the whole colour frame, 3 x N.
        u, v = np.meshgrid(np.arange(w_c), np.arange(h_c))
        pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).T
        rays = np.linalg.inv(self.K_color) @ pix

        # Scale rays to depth, rotate and translate into the thermal frame,
        # then project and divide through by z.
        proj = self.K_thermal @ (self.R @ (rays * self.depth_baselines) + self.t)
        proj /= proj[2, :]
        u_t = proj[0, :].reshape(h_c, w_c).astype(np.float32)
        v_t = proj[1, :].reshape(h_c, w_c).astype(np.float32)

        warped = cv2.remap(thermal, u_t, v_t, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Colour pixels whose ray lands outside the thermal sensor have no
        # data -- the thermal FOV is narrower than the colour FOV. The upper
        # bound is the last pixel centre, not the sensor edge: between the
        # two, bilinear sampling blends in the zero border value.
        valid = (u_t >= 0) & (u_t <= w_t - 1) & (v_t >= 0) & (v_t <= h_t - 1)
        return warped, valid


    def _thermal_zscore(self, thermal, color_bgr, valid):
        """Standardize against vegetation. Sky and soil sit far from leaf
        temperature and would otherwise set the scale.

        Standardizing per frame also removes ambient drift between capture
        sessions, so a warm afternoon and a cool morning become comparable.
        Returns None when the frame has too little vegetation to reference.
        """
        b, g, r = cv2.split(color_bgr.astype(np.float32))
        green = (g > b) & (g > r) & valid
        if green.sum() < self.min_green_pixels:
            green = valid          # fall back to the whole valid region
        if green.sum() < self.min_green_pixels:
            return None

        values = thermal[green].astype(np.float32)
        std = values.std()
        if std < 1e-6:             # uniform frame, nothing to normalize
            return None
        return (thermal.astype(np.float32) - values.mean()) / std

    @staticmethod
    def _letterbox(x, size, fill=0.0):
        """Scale to fit, then centre-pad to a square.

        Returns the scale and padding so boxes can be moved by the same
        transform. Stretching to a square instead would distort every object
        by the frame's aspect ratio.
        """
        _, h, w = x.shape
        ratio = min(size / w, size / h)
        nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
        x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear",
                          align_corners=False, antialias=True).squeeze(0)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        x = F.pad(x, (pad_x, size - nw - pad_x, pad_y, size - nh - pad_y),
                  value=fill)
        return x, ratio, pad_x, pad_y

    def _depth_relief(self, depth, color_bgr, valid, scale=100.0, clip=10.0):
        """Depth relative to the vegetation median, in fixed units.

        Unlike _depth_zscore this does not divide by the vegetation spread.
        Depth units are already consistent between frames, and the spread
        reflects canopy geometry, so dividing by it would give the same leaf
        curl a different size in different frames. scale converts sensor
        units to output units (100 turns millimetres into 10 cm steps); clip
        bounds rows far behind the canopy so they cannot dominate.
        """
        b, g, r = cv2.split(color_bgr.astype(np.float32))
        green = (g > b) & (g > r) & valid & (depth > 0)
        if green.sum() < self.min_green_pixels:
            green = valid & (depth > 0)
        if green.sum() < self.min_green_pixels:
            return None

        reference = np.median(depth[green].astype(np.float32))
        out = depth.astype(np.float32)
        out[depth <= 0] = reference             # holes become canopy level
        return np.clip((out - reference) / scale, -clip, clip)
    
    # ---- main path -----------------------------------------------------

    def __getitem__(self, index):
        root, img_id = self.samples[index]
        color_path, thermal_path, box_path, depth_path = self._paths(root, img_id)

        # 1. Load. A missing or unreadable file drops the frame rather than
        #    raising, so one bad file does not kill a training run.
        color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        thermal_raw = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        if color_bgr is None or thermal_raw is None:
            return None
        depth_raw = None
        if self.use_depth:
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                return None

        # 2. Register thermal onto colour.
        warped, valid = self._warp_thermal(thermal_raw, color_bgr.shape[:2])
        if not valid.any():
            return None

        # 3. Crop to the region covered by both sensors. Everything after
        #    this point lives in crop coordinates.
        ys, xs = np.where(valid)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        color_bgr = color_bgr[y0:y1, x0:x1]
        warped = warped[y0:y1, x0:x1]
        valid = valid[y0:y1, x0:x1]
        if depth_raw is not None:
            depth_raw = depth_raw[y0:y1, x0:x1]

        # 4. Raw counts become a comparable scale across frames.
        thermal = self._thermal_zscore(warped, color_bgr, valid)
        if thermal is None:
            return None
        depth = None
        if depth_raw is not None:
            depth = self._depth_relief(depth_raw, color_bgr, valid)
            # depth = self._depth_zscore(depth_raw, color_bgr, valid)
            if depth is None:
                return None

        # 5. Resolve class names to output ids, recording anything unmapped.
        boxes, names = self._load_boxes(box_path)
        keep_boxes, keep_labels = [], []
        for box, name in zip(boxes, names):
            label = self.class_map.get(name, None)
            if label is None:
                self.unmapped.add(name)     # surfaced by print_category_counts
                continue
            if label < 0:
                continue                    # dropped on purpose, stay quiet
            keep_boxes.append(box)
            keep_labels.append(label)

        boxes = np.array(keep_boxes, np.float32).reshape(-1, 4)
        labels = np.array(keep_labels, np.int64)

        # 6. Boxes are in full-frame coordinates; shift into the crop and
        #    clip. A box entirely outside becomes zero-area and is dropped
        #    by the min_side test in step 9.
        h, w = color_bgr.shape[:2]
        if len(boxes):
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        color = torch.from_numpy(
            cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        thermal = torch.from_numpy(thermal).unsqueeze(0)
        if depth is not None:
            depth = torch.from_numpy(depth).unsqueeze(0)

        # 7. Flips are exact -- no interpolation, so boxes stay precise.
        #    Every modality flips together or registration is destroyed.
        if self.augment:
            if torch.rand(1).item() < 0.5:
                color, thermal = torch.flip(color, [2]), torch.flip(thermal, [2])
                if depth is not None:
                    depth = torch.flip(depth, [2])
                if len(boxes):
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if torch.rand(1).item() < 0.5:
                color, thermal = torch.flip(color, [1]), torch.flip(thermal, [1])
                if depth is not None:
                    depth = torch.flip(depth, [1])
                if len(boxes):
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        # 8. Every modality shares one letterbox transform, so they stay
        #    aligned and one set of box coordinates serves all. Thermal and
        #    depth pad with 0, which after the z-score is the vegetation mean
        #    rather than an out-of-distribution constant.
        color, ratio, pad_x, pad_y = self._letterbox(color, self.size)
        thermal, _, _, _ = self._letterbox(thermal, self.size)
        color = (color - IMAGENET_MEAN) / IMAGENET_STD
        thermal = thermal.repeat(3, 1, 1)   # matches how SSL consumed thermal
        if depth is not None:
            depth, _, _, _ = self._letterbox(depth, self.size)
            depth = depth.repeat(3, 1, 1)

        # 9. Move boxes through the same transform and emit both formats:
        #    normalized cxcywh for the DETR loss, absolute xyxy for eval.
        xyxy, cxcywh, kept = [], [], []
        for (bx0, by0, bx1, by1), label in zip(boxes, labels):
            bx0, bx1 = bx0 * ratio + pad_x, bx1 * ratio + pad_x
            by0, by1 = by0 * ratio + pad_y, by1 * ratio + pad_y
            if bx1 - bx0 <= self.min_side or by1 - by0 <= self.min_side:
                continue
            xyxy.append([bx0, by0, bx1, by1])
            cxcywh.append([(bx0 + bx1) / 2 / self.size,
                           (by0 + by1) / 2 / self.size,
                           (bx1 - bx0) / self.size, (by1 - by0) / self.size])
            kept.append(int(label))

        record = {
            "pixel_values": color,
            "thermal_values": thermal,
            "labels": {
                "class_labels": torch.tensor(kept, dtype=torch.long),
                "boxes": torch.tensor(cxcywh, dtype=torch.float32).reshape(-1, 4),
                "boxes_xyxy": torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
                "image_id": img_id,
            },
        }
        if depth is not None:
            record["depth_values"] = depth
        return record

    # ---- reporting -----------------------------------------------------

    def num_classes(self):
        kept = [v for v in self.class_map.values() if v >= 0]
        return max(kept) + 1 if kept else 0

    def print_category_counts(self):
        """Box and image counts per class, plus any names the class_map missed.

        Walks the whole dataset, so it decodes and warps every frame. Slow,
        but it reports what training will actually see rather than what the
        annotation files contain.
        """
        boxes, images, dropped = {}, {}, 0
        for i in range(len(self)):
            record = self[i]
            if record is None:
                dropped += 1
                continue
            seen = set()
            for label in record["labels"]["class_labels"].tolist():
                boxes[label] = boxes.get(label, 0) + 1
                seen.add(label)
            for label in seen:
                images[label] = images.get(label, 0) + 1

        source = {}
        for name, label in self.class_map.items():
            if label >= 0:
                source.setdefault(label, []).append(name)

        print(f"{len(self)} samples, {dropped} unusable, "
              f"{sum(boxes.values())} boxes")
        for label in sorted(boxes):
            print(f"    class {label:<4}{boxes[label]:>7} boxes"
                  f"{images[label]:>7} images   "
                  f"{', '.join(sorted(source.get(label, [])))}")
        if self.unmapped:
            print(f"    names seen but not in class_map: "
                  f"{sorted(self.unmapped)}")
        return boxes


def collate_fn(batch):
    """Stack a batch, discarding frames __getitem__ rejected.

    Returns None when every frame in the batch was rejected, so the training
    loop needs a `if batch is None: continue` guard.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "thermal_values": torch.stack([b["thermal_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=str, required=True,
                   help="dataset list file, '<root> <img_id>' per line")
    p.add_argument("--calibration", type=str, required=True,
                   help="camera calibration json")
    p.add_argument("--classes", type=str, nargs="+", required=True,
                   help="class names to keep, all merged into one id")
    p.add_argument("--size", type=int, default=576,
                   help="letterboxed square output resolution")
    p.add_argument("--depth-baselines", type=float, default=7.0,
                   help="constant warp depth, in calibration baselines")
    args = p.parse_args()

    dataset = MultiSpectralDetection(
        args.samples, args.calibration,
        class_map={name: 0 for name in args.classes},
        size=args.size, depth_baselines=args.depth_baselines)
    dataset.print_category_counts()


if __name__ == "__main__":
    main()