#!/usr/bin/env python3
"""Binary leaf classifier over one or more modalities, end to end.

The linear probe measured what frozen DINO features already encode. This
trains the feature extractors themselves: one RF-DETR DINOv2 tower per
modality, initialized from the same weights the probe used, their pooled
outputs concatenated into an MLP head.

Crops come from the same boxes the probe uses -- positive classes against
negative classes, with negatives that overlap a positive discarded -- so the
numbers are directly comparable to the probe's.

--unfreeze controls how much of each tower trains. 0 reproduces the probe's
frozen features with a nonlinear head; 2 to 4 is the usual range at this data
volume; -1 trains everything and will almost certainly overfit.
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import RfDetrForObjectDetection

from multispectral_detection_loader import MultiSpectralDetection

CHECKPOINT = "Roboflow/rf-detr-medium"
POSITIVE = 0
NEGATIVE = 1
MODALITY_KEYS = {"color": "pixel_values",
                 "thermal": "thermal_values",
                 "depth": "depth_values"}


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def crop_window(image, x, y, side, size):
    _, h, w = image.shape
    a = int(max(0, min(x, w - side)))
    b = int(max(0, min(y, h - side)))
    window = image[:, b:b + int(side), a:a + int(side)]
    if window.shape[1] < 2 or window.shape[2] < 2:
        return None
    return nn.functional.interpolate(
        window.unsqueeze(0), size=(size, size),
        mode="bilinear", align_corners=False).squeeze(0)


def build_crops(dataset, modalities, size, pad, max_negatives, seed):
    """Extract every crop once, up front.

    Crops are small and the dataset is not, so holding them in memory costs
    far less than re-warping and re-cropping each frame every epoch.
    """
    rng = random.Random(seed)
    samples, skipped, dropped = [], 0, 0

    for index in range(len(dataset)):
        record = dataset[index]
        if record is None:
            skipped += 1
            continue

        images = {m: record[MODALITY_KEYS[m]] for m in modalities}
        boxes = record["labels"]["boxes_xyxy"].numpy().reshape(-1, 4)
        classes = record["labels"]["class_labels"].tolist()

        positive = [b for b, l in zip(boxes, classes) if l == POSITIVE]
        negative = [b for b, l in zip(boxes, classes) if l == NEGATIVE]
        clean = [n for n in negative if not any(overlaps(n, p) for p in positive)]
        dropped += len(negative) - len(clean)
        if max_negatives and len(clean) > max_negatives:
            rng.shuffle(clean)
            clean = clean[:max_negatives]

        for box, label in ([(b, POSITIVE) for b in positive]
                           + [(b, NEGATIVE) for b in clean]):
            x0, y0, x1, y1 = box
            side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
            x, y = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
            cut = {m: crop_window(images[m], x, y, side, size)
                   for m in modalities}
            if any(v is None for v in cut.values()):
                continue
            samples.append((cut, label))

    labels = np.array([s[1] for s in samples])
    print(f"    {len(samples)} crops "
          f"({(labels == POSITIVE).sum()} positive, "
          f"{(labels == NEGATIVE).sum()} negative), "
          f"{dropped} negatives dropped for overlap, "
          f"{skipped} frames skipped")
    if not samples:
        raise SystemExit("no crops extracted")
    return samples


class CropDataset(Dataset):
    """Holds pre-extracted crops; flips are the only augmentation.

    Flips apply to every modality together, or the registration between them
    is destroyed.
    """

    def __init__(self, samples, modalities, augment=False):
        self.samples = samples
        self.modalities = modalities
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        crops, label = self.samples[index]
        crops = {m: crops[m] for m in self.modalities}
        if self.augment:
            if torch.rand(1).item() < 0.5:
                crops = {m: torch.flip(v, [2]) for m, v in crops.items()}
            if torch.rand(1).item() < 0.5:
                crops = {m: torch.flip(v, [1]) for m, v in crops.items()}
        return crops, label


def collate_fn(batch):
    modalities = batch[0][0].keys()
    return ({m: torch.stack([b[0][m] for b in batch]) for m in modalities},
            torch.tensor([b[1] for b in batch], dtype=torch.long))


class Tower(nn.Module):
    """One RF-DETR DINOv2 backbone, pooled to a vector per crop."""

    def __init__(self, unfreeze, unfreeze_from="top"):
        super().__init__()
        detector = RfDetrForObjectDetection.from_pretrained(CHECKPOINT)
        self.backbone = detector.model.backbone.backbone
        self.dim = self.backbone.config.hidden_size
        self.set_trainable(unfreeze, unfreeze_from)

    def set_trainable(self, unfreeze, unfreeze_from):
        """unfreeze counts transformer blocks to train; -1 trains all.

        Direction matters and depends on what is mismatched. "top" trains the
        last blocks, which is the usual recipe when the imagery is familiar
        and only the task is new. "bottom" trains the stem and first blocks,
        which is what a modality shift calls for: a z-scored thermal or depth
        field has edge and intensity statistics unlike natural RGB, and that
        is read by the early layers, not the late ones.
        """
        if unfreeze < 0:
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze == 0:
            return

        layers = self.backbone.encoder.layer
        chosen = layers[:unfreeze] if unfreeze_from == "bottom" \
            else layers[-unfreeze:]
        for layer in chosen:
            for p in layer.parameters():
                p.requires_grad = True

        if unfreeze_from == "bottom":
            # The patch embedding is where the channel statistics first land,
            # so it trains alongside the early blocks.
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = True
        elif hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        feats = getattr(out, "feature_maps", None)
        if feats is None:
            feats = getattr(out, "last_hidden_state", out)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.dim() == 4:
            return feats.mean(dim=(2, 3))
        return feats.mean(dim=1)


class LeafClassifier(nn.Module):
    """One tower per modality, concatenated into an MLP head."""

    def __init__(self, modalities, unfreeze, unfreeze_from="top",
                 hidden=512, dropout=0.3):
        super().__init__()
        self.modalities = modalities
        self.towers = nn.ModuleDict(
            {m: Tower(unfreeze, unfreeze_from) for m in modalities})
        width = sum(t.dim for t in self.towers.values())
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, crops):
        # Single-channel modalities are replicated to match the stem, as in
        # the probe; the replication carries no extra information.
        feats = []
        for m in self.modalities:
            x = crops[m]
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            feats.append(self.towers[m](x))
        return self.head(torch.cat(feats, dim=1))


def evaluate(model, loader, device):
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for crops, labels in loader:
            crops = {m: v.to(device) for m, v in crops.items()}
            logits = model(crops)
            pred.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())
    return np.array(truth), np.array(pred)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-samples", type=str, required=True,
                   help="dataset list file for training")
    p.add_argument("--eval-samples", type=str, required=True,
                   help="dataset list file for evaluation")
    p.add_argument("--calibration", type=str, required=True,
                   help="camera calibration json for the thermal warp")
    p.add_argument("--box-dir", type=str, default="combined_roboflow_sam3",
                   help="subdirectory under each root holding box pickles")
    p.add_argument("--positive-classes", type=str, nargs="+", required=True,
                   help="box class names forming the positive class")
    p.add_argument("--negative-classes", type=str, nargs="+", required=True,
                   help="box class names forming the negative class; any that "
                        "overlap a positive box are discarded")
    p.add_argument("--modalities", type=str, nargs="+", default=["color"],
                   choices=["color", "thermal", "depth"],
                   help="one tower is built per modality")
    p.add_argument("--unfreeze", type=int, default=2,
                   help="transformer blocks to train per tower; "
                        "0 freezes the towers, -1 trains everything")
    p.add_argument("--unfreeze-from", type=str, default="top",
                   choices=["top", "bottom"],
                   help="which end to unfreeze: top for task shift, bottom "
                        "for modality shift (also trains the patch embedding)")
    p.add_argument("--epochs", type=int, default=30,
                   help="number of training epochs")
    p.add_argument("--batch-size", type=int, default=32,
                   help="crops per batch")
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="learning rate for unfrozen backbone blocks")
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="learning rate for the classifier head")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay")
    p.add_argument("--hidden", type=int, default=512,
                   help="width of the hidden layer in the head")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="dropout in the head")
    p.add_argument("--max-negatives", type=int, default=8,
                   help="cap on negative boxes kept per frame; 0 keeps all")
    p.add_argument("--size", type=int, default=224,
                   help="crop resolution fed to the towers")
    p.add_argument("--pad", type=float, default=0.1,
                   help="context margin around each box")
    p.add_argument("--frame-size", type=int, default=576,
                   help="letterboxed frame size from the loader")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for negative sampling and initialization")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker processes")
    p.add_argument("--results", type=str, default="",
                   help="CSV file to append one row of metrics to")
    p.add_argument("--save", type=str, default="",
                   help="path to save the best checkpoint to")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    class_map = {name: POSITIVE for name in args.positive_classes}
    for name in args.negative_classes:
        class_map[name] = NEGATIVE

    use_depth = "depth" in args.modalities
    common = dict(size=args.frame_size, use_depth=use_depth,
                  box_dir=args.box_dir)
    train_set = MultiSpectralDetection(args.train_samples, args.calibration,
                                       class_map, **common)
    eval_set = MultiSpectralDetection(args.eval_samples, args.calibration,
                                      class_map, **common)

    print(f"modalities: {'+'.join(args.modalities)}  "
          f"unfreeze: {args.unfreeze} from {args.unfreeze_from}")
    print(f"positive: {', '.join(args.positive_classes)}")
    print(f"negative: {', '.join(args.negative_classes)}")

    print("extracting train crops")
    train_crops = build_crops(train_set, args.modalities, args.size, args.pad,
                              args.max_negatives, args.seed)
    print("extracting eval crops")
    eval_crops = build_crops(eval_set, args.modalities, args.size, args.pad,
                             args.max_negatives, args.seed + 1)

    train_loader = DataLoader(
        CropDataset(train_crops, args.modalities, augment=True),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=args.workers)
    eval_loader = DataLoader(
        CropDataset(eval_crops, args.modalities, augment=False),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeafClassifier(args.modalities, args.unfreeze, args.unfreeze_from,
                           args.hidden, args.dropout).to(device)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("towers")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("head")]
    print(f"{sum(p.numel() for p in backbone_params):,} backbone params, "
          f"{sum(p.numel() for p in head_params):,} head params")

    groups = [{"params": head_params, "lr": args.head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=args.epochs)

    # Negatives outnumber positives, so an unweighted loss is minimized by
    # answering negative. Weight inversely to frequency.
    counts = np.bincount([s[1] for s in train_crops], minlength=2)
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    print(f"class weights: positive {weights[0]:.3f}, negative {weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    target_names = ["+".join(args.positive_classes), "negative"]
    best_f1, best_epoch = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for crops, labels in train_loader:
            crops = {m: v.to(device) for m, v in crops.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(crops), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        schedule.step()

        truth, pred = evaluate(model, eval_loader, device)
        report = classification_report(truth, pred, labels=[POSITIVE, NEGATIVE],
                                       target_names=target_names,
                                       output_dict=True, zero_division=0)
        pos_f1 = report[target_names[0]]["f1-score"]
        print(f"epoch {epoch + 1:>3}/{args.epochs}  "
              f"loss {running / len(train_loader):.4f}  "
              f"positive F1 {pos_f1:.4f}")

        if pos_f1 > best_f1:
            best_f1, best_epoch = pos_f1, epoch + 1
            best_report, best_truth, best_pred = report, truth, pred
            if args.save:
                torch.save({"epoch": epoch + 1, "positive_f1": pos_f1,
                            "modalities": args.modalities,
                            "model_state_dict": model.state_dict()}, args.save)

    print(f"\n--- best epoch {best_epoch}, positive F1 {best_f1:.4f} ---")
    print(classification_report(best_truth, best_pred,
                                labels=[POSITIVE, NEGATIVE],
                                target_names=target_names, digits=4,
                                zero_division=0))
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':<20}" + "".join(f"{n[:14]:>16}" for n in target_names))
    for name, row in zip(target_names,
                         confusion_matrix(best_truth, best_pred,
                                          labels=[POSITIVE, NEGATIVE])):
        print(f"{name:<20}" + "".join(f"{v:>16}" for v in row))

    if args.results:
        pos = best_report[target_names[0]]
        neg = best_report["negative"]
        fields = ["positive_classes", "negative_classes", "modalities",
                  "unfreeze", "unfreeze_from", "backbone_lr", "head_lr",
                  "epochs", "best_epoch", "seed", "max_negatives",
                  "n_train", "n_eval", "support_pos", "support_neg",
                  "pos_precision", "pos_recall", "pos_f1",
                  "neg_precision", "neg_recall", "neg_f1",
                  "accuracy", "macro_f1", "weighted_f1"]
        row = {
            "positive_classes": "+".join(args.positive_classes),
            "negative_classes": "+".join(args.negative_classes),
            "modalities": "+".join(args.modalities),
            "unfreeze": args.unfreeze,
            "unfreeze_from": args.unfreeze_from,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "max_negatives": args.max_negatives,
            "n_train": len(train_crops),
            "n_eval": len(eval_crops),
            "support_pos": int(pos["support"]),
            "support_neg": int(neg["support"]),
            "pos_precision": round(pos["precision"], 4),
            "pos_recall": round(pos["recall"], 4),
            "pos_f1": round(pos["f1-score"], 4),
            "neg_precision": round(neg["precision"], 4),
            "neg_recall": round(neg["recall"], 4),
            "neg_f1": round(neg["f1-score"], 4),
            "accuracy": round(best_report["accuracy"], 4),
            "macro_f1": round(best_report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(best_report["weighted avg"]["f1-score"], 4),
        }
        exists = os.path.exists(args.results)
        with open(args.results, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nappended to {args.results}")


if __name__ == "__main__":
    main()