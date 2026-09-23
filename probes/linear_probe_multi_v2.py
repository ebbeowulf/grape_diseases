#!/usr/bin/env python3
"""Linear probe on crops from the multi-spectral loader.

Asks whether frozen features separate the target class from background, with
no detector in the loop. Crops come from the same warped, cropped, z-scored
tensors the detector trains on.

Pass one or more --modalities from {color, thermal, depth}. With several:

    --fusion late   each modality gets its own encoder and the features are
                    concatenated. Keeps both representations whole, discards
                    spatial correspondence between them.
    --fusion early  channels are stacked into one encoder whose stem is
                    widened, new channels initialized to the RGB kernel mean.
                    Correspondence survives, but the stem is no longer
                    exactly the pretrained one -- a real handicap while the
                    backbone is frozen.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import RfDetrForObjectDetection

from multispectral_detection_loader import MultiSpectralDetection

CHECKPOINT = "Roboflow/rf-detr-medium"
POSITIVE = 0
NEGATIVE = 1
STEM_PATH = "embeddings.patch_embeddings.projection"


def get_module(root, dotted):
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


def set_module(root, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        root = getattr(root, part)
    setattr(root, parts[-1], value)


def retarget_stem(backbone, in_channels):
    """Rebuild the patch embedding for a different channel count.

    Fewer than 3: sum the kernels. Convolving a replicated channel equals
    convolving once with the channel sum, so collapsing to 1 is exact.
    More than 3: append kernels initialized to the RGB mean, which starts
    each new channel behaving like a grayscale copy of the others.
    """
    conv = get_module(backbone, STEM_PATH)
    if conv.in_channels == in_channels:
        return

    weight = conv.weight.data
    if in_channels == 1:
        new_weight = weight.sum(dim=1, keepdim=True)
    elif in_channels > 3:
        extra = weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
        new_weight = torch.cat([weight, extra], dim=1)
    else:
        raise SystemExit(f"no stem rule for {in_channels} channels")

    new_conv = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         bias=conv.bias is not None)
    new_conv.weight.data = new_weight
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    set_module(backbone, STEM_PATH, new_conv)

    backbone.config.num_channels = in_channels
    patch_embed = get_module(backbone, "embeddings.patch_embeddings")
    if hasattr(patch_embed, "num_channels"):
        patch_embed.num_channels = in_channels


class Encoder:
    """RF-DETR's backbone, frozen, pooled to one vector per crop."""

    def __init__(self, weights, ssl_checkpoint, device, in_channels=3):
        detector = RfDetrForObjectDetection.from_pretrained(CHECKPOINT)
        self.backbone = detector.model.backbone.backbone

        if weights == "ssl":
            if not ssl_checkpoint:
                raise SystemExit("--ssl-checkpoint is required for ssl weights")
            self.backbone.load_state_dict(
                torch.load(ssl_checkpoint, map_location="cpu"), strict=True)
            print(f"    loaded SSL backbone: {ssl_checkpoint}")
        elif weights == "random":
            torch.manual_seed(0)
            try:
                self.backbone.apply(self.backbone._init_weights)
            except AttributeError:
                for m in self.backbone.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
            print("    random init: no pretrained weights")
        else:
            print("    stock RF-DETR backbone")

        # Surgery comes after loading, so it applies to the loaded weights.
        retarget_stem(self.backbone, in_channels)
        self.backbone.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, crops):
        out = self.backbone(crops.to(self.device))
        x = getattr(out, "feature_maps", None)
        if x is None:
            x = getattr(out, "last_hidden_state", out)
        if isinstance(x, (tuple, list)):
            x = x[-1]
        if x.dim() == 4:
            return x.mean(dim=(2, 3)).float().cpu()
        return x.mean(dim=1).float().cpu()


def crop_window(image, x, y, side, size):
    _, h, w = image.shape
    a = int(max(0, min(x, w - side)))
    b = int(max(0, min(y, h - side)))
    window = image[:, b:b + int(side), a:a + int(side)]
    if window.shape[1] < 2 or window.shape[2] < 2:
        return None
    return F.interpolate(window.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze(0)


def box_windows(boxes, pad):
    """Square window geometry per box, shared by both modalities so the
    colour and thermal crops cover identical pixels."""
    out = []
    for x0, y0, x1, y1 in boxes:
        side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        out.append(((x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side))
    return out


def overlaps(a, b):
    """True when two xyxy boxes share any area at all."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def split_boxes(boxes, labels, max_negatives, rng):
    """Positive boxes, and negative boxes that touch none of them.

    A leaf overlapping a lesion is ambiguous -- it contains the thing we are
    trying to detect -- so it is discarded rather than labelled either way.
    Negatives are capped per frame because a frame can hold far more leaves
    than lesions, and an unbounded ratio would let the classifier win by
    always answering negative.
    """
    positive = [b for b, l in zip(boxes, labels) if l == POSITIVE]
    negative = [b for b, l in zip(boxes, labels) if l == NEGATIVE]

    clean = [n for n in negative if not any(overlaps(n, p) for p in positive)]
    dropped = len(negative) - len(clean)
    if max_negatives and len(clean) > max_negatives:
        rng.shuffle(clean)
        clean = clean[:max_negatives]
    return positive, clean, dropped


MODALITY_KEYS = {"color": "pixel_values",
                 "thermal": "thermal_values",
                 "depth": "depth_values"}
MODALITY_CHANNELS = {"color": 3, "thermal": 1, "depth": 1}


def extract(dataset, modalities, fusion, encoders, size, pad,
            max_negatives, seed):
    rng = random.Random(seed)
    features, labels, skipped, dropped = [], [], 0, 0

    for index in range(len(dataset)):
        record = dataset[index]
        if record is None:
            skipped += 1
            continue

        images = {m: record[MODALITY_KEYS[m]] for m in modalities}
        boxes = record["labels"]["boxes_xyxy"].numpy().reshape(-1, 4)
        classes = record["labels"]["class_labels"].tolist()

        positive, negative, n_dropped = split_boxes(boxes, classes,
                                                    max_negatives, rng)
        dropped += n_dropped
        if not positive and not negative:
            continue

        # One window list drives every modality, so the crops are registered.
        windows = box_windows(positive + negative, pad)
        window_labels = [POSITIVE] * len(positive) + [NEGATIVE] * len(negative)

        crops = {m: [] for m in modalities}
        kept = []
        for (x, y, side), label in zip(windows, window_labels):
            cut = {m: crop_window(images[m], x, y, side, size)
                   for m in modalities}
            if any(v is None for v in cut.values()):
                continue
            for m in modalities:
                # Modalities stored replicated carry no extra information.
                crops[m].append(cut[m][:MODALITY_CHANNELS[m]])
            kept.append(label)

        if not kept:
            continue
        stacked = {m: torch.stack(crops[m]) for m in modalities}

        if fusion == "early":
            feats = encoders["fused"](torch.cat(
                [stacked[m] for m in modalities], dim=1))
        else:
            feats = torch.cat([encoders[m](
                stacked[m].repeat(1, 3 // MODALITY_CHANNELS[m], 1, 1))
                for m in modalities], dim=1)

        features.append(feats)
        labels.extend(kept)

    if not features:
        raise SystemExit("no crops extracted")
    labels = np.array(labels)
    print(f"    {len(labels)} crops "
          f"({(labels == POSITIVE).sum()} positive, "
          f"{(labels == NEGATIVE).sum()} negative), "
          f"{dropped} negatives dropped for overlap, "
          f"{skipped} frames skipped")
    return torch.cat(features).numpy(), labels


def build_encoders(modalities, fusion, args, device):
    """Late fusion gives each modality its own encoder; early fusion stacks
    channels into one whose stem is widened to match."""
    encoders = {}
    if fusion == "early":
        channels = sum(MODALITY_CHANNELS[m] for m in modalities)
        print(f"  fused encoder ({channels} channels):")
        encoders["fused"] = Encoder(args.color_weights, "", device, channels)
        return encoders

    for m in modalities:
        print(f"  {m} encoder:")
        weights = {"color": args.color_weights,
                   "thermal": args.thermal_weights,
                   "depth": args.depth_weights}[m]
        checkpoint = args.ssl_checkpoint if m == "thermal" else ""
        encoders[m] = Encoder(weights, checkpoint, device, 3)
    return encoders


FIELDS = ["positive_classes", "negative_classes", "modalities", "fusion",
          "color_weights", "thermal_weights", "depth_weights",
          "max_negatives", "pad", "size",
          "n_train", "n_eval", "support_pos", "support_bg",
          "pos_precision", "pos_recall", "pos_f1",
          "bg_precision", "bg_recall", "bg_f1",
          "accuracy", "macro_f1", "weighted_f1"]


def append_result(path, row):
    """One row per run, appended. Header written only when creating the file,
    so a sweep can keep adding to the same CSV across sessions."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-samples", type=str, required=True,
                   help="dataset list file for fitting the probe")
    p.add_argument("--eval-samples", type=str, required=True,
                   help="dataset list file for reporting")
    p.add_argument("--calibration", type=str, required=True,
                   help="camera calibration json for the thermal warp")
    p.add_argument("--positive-classes", type=str, nargs="+", required=True,
                   help="box class names forming the positive class")
    p.add_argument("--negative-classes", type=str, nargs="+", required=True,
                   help="box class names forming the negative class; any that "
                        "overlap a positive box are discarded")
    p.add_argument("--modalities", type=str, nargs="+", default=["color"],
                   choices=["color", "thermal", "depth"],
                   help="one or more modalities to probe")
    p.add_argument("--fusion", type=str, default="late",
                   choices=["late", "early"],
                   help="late concatenates per-modality features; early "
                        "stacks channels into one encoder")
    p.add_argument("--color-weights", type=str, default="stock",
                   choices=["stock", "random"],
                   help="initialization for the colour encoder")
    p.add_argument("--thermal-weights", type=str, default="stock",
                   choices=["stock", "ssl", "random"],
                   help="initialization for the thermal encoder")
    p.add_argument("--ssl-checkpoint", type=str, default="",
                   help="thermal backbone state_dict; required for ssl weights")
    p.add_argument("--depth-weights", type=str, default="stock",
                   choices=["stock", "random"],
                   help="initialization for the depth encoder")
    p.add_argument("--results", type=str, default="",
                   help="CSV file to append one row of metrics to")
    p.add_argument("--max-negatives", type=int, default=8,
                   help="cap on negative boxes kept per frame; 0 keeps all")
    p.add_argument("--size", type=int, default=224,
                   help="crop resolution fed to the backbone")
    p.add_argument("--pad", type=float, default=0.1,
                   help="context margin around each box")
    p.add_argument("--frame-size", type=int, default=576,
                   help="letterboxed frame size from the loader")
    p.add_argument("--C", type=float, default=1.0,
                   help="inverse regularization strength")
    args = p.parse_args()

    class_map = {name: POSITIVE for name in args.positive_classes}
    for name in args.negative_classes:
        class_map[name] = NEGATIVE

    use_depth = "depth" in args.modalities
    train_set = MultiSpectralDetection(args.train_samples, args.calibration,
                                       class_map, size=args.frame_size,
                                       use_depth=use_depth,box_dir="roboflow_boxescombined_roboflow_sam3")
                                    #    use_depth=use_depth,box_dir="combined_roboflow_sam3")
    eval_set = MultiSpectralDetection(args.eval_samples, args.calibration,
                                      class_map, size=args.frame_size,
                                      use_depth=use_depth,box_dir="combined_roboflow_sam3")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = "+".join(args.modalities)
    print(f"modalities: {label} ({args.fusion} fusion)")
    print(f"positive: {', '.join(args.positive_classes)}")
    print(f"negative: {', '.join(args.negative_classes)}")
    encoders = build_encoders(args.modalities, args.fusion, args, device)

    print("extracting train crops")
    x_tr, y_tr = extract(train_set, args.modalities, args.fusion, encoders,
                         args.size, args.pad, args.max_negatives, seed=0)
    print("extracting eval crops")
    x_ev, y_ev = extract(eval_set, args.modalities, args.fusion, encoders,
                         args.size, args.pad, args.max_negatives, seed=1)

    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(C=args.C, max_iter=3000, random_state=0)
    clf.fit(scaler.transform(x_tr), y_tr)
    pred = clf.predict(scaler.transform(x_ev))

    labels = [POSITIVE, NEGATIVE]
    positive_name = "+".join(args.positive_classes)
    names = {POSITIVE: positive_name, NEGATIVE: "negative"}
    target_names = [names[c] for c in labels]

    print(f"\n--- {label} | {args.fusion} fusion | evaluation set ---")
    print(classification_report(y_ev, pred, labels=labels,
                                target_names=target_names, digits=4,
                                zero_division=0))
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':<20}" + "".join(f"{n[:14]:>16}" for n in target_names))
    for name, row in zip(target_names, confusion_matrix(y_ev, pred, labels=labels)):
        print(f"{name:<20}" + "".join(f"{v:>16}" for v in row))

    if args.results:
        report = classification_report(y_ev, pred, labels=labels,
                                       target_names=target_names,
                                       output_dict=True, zero_division=0)
        positive = report.get(positive_name, {})
        bg = report.get("negative", {})
        append_result(args.results, {
            "positive_classes": positive_name,
            "negative_classes": "+".join(args.negative_classes),
            "modalities": label,
            "fusion": args.fusion,
            "color_weights": args.color_weights,
            "thermal_weights": args.thermal_weights,
            "depth_weights": args.depth_weights,
            "max_negatives": args.max_negatives,
            "pad": args.pad,
            "size": args.size,
            "n_train": len(y_tr),
            "n_eval": len(y_ev),
            "support_pos": int(positive.get("support", 0)),
            "support_bg": int(bg.get("support", 0)),
            "pos_precision": round(positive.get("precision", 0.0), 4),
            "pos_recall": round(positive.get("recall", 0.0), 4),
            "pos_f1": round(positive.get("f1-score", 0.0), 4),
            "bg_precision": round(bg.get("precision", 0.0), 4),
            "bg_recall": round(bg.get("recall", 0.0), 4),
            "bg_f1": round(bg.get("f1-score", 0.0), 4),
            "accuracy": round(report.get("accuracy", 0.0), 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        })
        print(f"\nappended to {args.results}")


if __name__ == "__main__":
    main()