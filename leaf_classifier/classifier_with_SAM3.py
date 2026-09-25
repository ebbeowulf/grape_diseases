#!/usr/bin/env python3
"""Binary leaf classifier over SAM3 leaf proposals, end to end.

SAM3 acts as the proposal network. Its "healthy leaf" and "unhealthy leaf"
boxes are merged with NMS across prompts, and each surviving proposal is
labelled by its best IoU against the ground-truth boxes of the target
disease: above --pos-iou it is positive. In training, proposals at or below
--neg-iou are negative and those in between are ignored; in evaluation there
is no ignore band -- at or below --pos-iou is negative.

Two evaluations are reported:
    per leaf   each proposal is a sample; every GT box with no proposal
               above --pos-iou is added as a false negative, so the score
               covers SAM3 and the classifier together
    per image  a frame is truly positive if it holds any target GT box, and
               predicted positive if at least --image-min-positive of its
               proposals are classified positive

One RF-DETR DINOv2 tower is trained per modality, pooled outputs
concatenated into an MLP head. --unfreeze controls how much of each tower
trains: 0 freezes it, -1 trains everything.

--gt-positives 1 also trains on crops of the GT boxes themselves, as extra
positives. Evaluation is always on proposals only.
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import RfDetrForObjectDetection

from multispectral_detection_loader import MultiSpectralDetection

CHECKPOINT = "Roboflow/rf-detr-medium"

# Loader class ids: target-disease GT boxes and SAM3 leaf proposals. Any
# other name (other diseases) is dropped by the loader.
GT = 0
PROPOSAL = 1

# Classifier labels.
POSITIVE = 0
NEGATIVE = 1

MODALITY_KEYS = {"color": "pixel_values",
                 "thermal": "thermal_values",
                 "depth": "depth_values"}


# ---- box geometry ----------------------------------------------------------

def box_iou(a, b):
    """IoU matrix between xyxy arrays a (N, 4) and b (M, 4)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes, scores, threshold):
    """Greedy NMS; returns kept indices, highest score first.

    A box is suppressed when its IoU with a kept box exceeds threshold.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        overlap = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return np.array(keep, dtype=np.int64)


# ---- crop extraction -------------------------------------------------------

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


def crop_box(images, box, size, pad):
    """One square crop per modality around box, or None if any is degenerate."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    x, y = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
    cut = {}
    for m, image in images.items():
        window = crop_window(image, x, y, side, size)
        if window is None:
            return None
        cut[m] = window.numpy()
    return cut


class FrameProposals(Dataset):
    """Loads one frame, merges its proposals and crops them, in a worker.

    Returns the crops with each proposal's best IoU against the target GT
    boxes, each GT box's best IoU against the proposals, and -- when
    gt_crops is set -- crops of the GT boxes themselves. Labels are not
    assigned here: thresholds are applied in the main process, so extraction
    does not depend on them.

    Thermal and depth arrive replicated to three identical channels; one is
    kept and the model replicates it back. Crops come back as numpy arrays,
    since tensors returned from workers each pin a shared-memory descriptor.
    """

    def __init__(self, dataset, modalities, size, pad, nms_iou, gt_crops=False):
        self.dataset = dataset
        self.modalities = modalities
        self.size = size
        self.pad = pad
        self.nms_iou = nms_iou
        self.gt_crops = gt_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset[index]
        if record is None:
            return None

        images = {m: record[MODALITY_KEYS[m]] for m in self.modalities}
        images = {m: v if m == "color" else v[:1] for m, v in images.items()}
        boxes = record["labels"]["boxes_xyxy"].numpy().reshape(-1, 4)
        classes = record["labels"]["class_labels"].numpy()
        scores = record["labels"]["scores"].numpy()

        gt = boxes[classes == GT]
        proposals = boxes[classes == PROPOSAL]
        n_raw = len(proposals)
        # Both prompts are merged before NMS, so one leaf found as healthy
        # and as unhealthy becomes a single proposal.
        proposals = proposals[nms(proposals, scores[classes == PROPOSAL],
                                  self.nms_iou)]

        cuts, kept = [], []
        for i, box in enumerate(proposals):
            cut = crop_box(images, box, self.size, self.pad)
            if cut is not None:
                cuts.append(cut)
                kept.append(i)
        proposals = proposals[kept]

        iou = box_iou(proposals, gt)
        proposal_iou = iou.max(axis=1) if len(gt) else np.zeros(len(proposals))
        gt_iou = iou.max(axis=0) if len(proposals) else np.zeros(len(gt))

        gt_cuts = []
        if self.gt_crops:
            gt_cuts = [c for c in (crop_box(images, box, self.size, self.pad)
                                   for box in gt) if c is not None]
        return cuts, proposal_iou, gt_iou, n_raw, gt_cuts


def unwrap(batch):
    return batch[0]


def quiet_worker(_):
    # PyTorch already limits each worker to one thread; OpenCV does not.
    cv2.setNumThreads(1)


def build_set(dataset, modalities, args, rng, train):
    """Extract and label every proposal crop once, up front.

    train=True applies the ignore band and the per-frame negative cap, and
    with --gt-positives adds each GT box's own crop as a positive.
    train=False keeps every proposal, labels at --pos-iou alone, and also
    returns each crop's frame and each frame's GT count and missed GT
    boxes, which the two evaluations need.
    """
    loader = DataLoader(
        FrameProposals(dataset, modalities, args.size, args.pad, args.nms_iou,
                       gt_crops=train and bool(args.gt_positives)),
        batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=unwrap, worker_init_fn=quiet_worker)

    cuts = {m: [] for m in modalities}
    labels, frames, gt_count, gt_missed = [], [], [], []
    skipped = ignored = raw = merged = from_gt = 0

    for item in loader:
        if item is None:
            skipped += 1
            continue
        crops, proposal_iou, gt_iou, n_raw, gt_cuts = item
        raw += n_raw
        merged += len(crops)

        positive = proposal_iou > args.pos_iou
        negative = proposal_iou <= (args.neg_iou if train else args.pos_iou)
        ignored += int((~positive & ~negative).sum())

        keep_neg = np.flatnonzero(negative).tolist()
        if train and args.max_negatives and len(keep_neg) > args.max_negatives:
            rng.shuffle(keep_neg)
            keep_neg = keep_neg[:args.max_negatives]

        frame = len(gt_count)
        for i, label in ([(i, POSITIVE) for i in np.flatnonzero(positive)]
                         + [(i, NEGATIVE) for i in keep_neg]):
            for m in modalities:
                cuts[m].append(crops[i][m])
            labels.append(label)
            frames.append(frame)
        for cut in gt_cuts:
            for m in modalities:
                cuts[m].append(cut[m])
            labels.append(POSITIVE)
            frames.append(frame)
        from_gt += len(gt_cuts)
        gt_count.append(len(gt_iou))
        gt_missed.append(int((gt_iou <= args.pos_iou).sum()))

    labels = np.array(labels, dtype=np.int64)
    gt_count = np.array(gt_count, dtype=np.int64)
    gt_missed = np.array(gt_missed, dtype=np.int64)
    print(f"    {len(gt_count)} frames used, {skipped} skipped; "
          f"{raw} proposals, {merged} after NMS")
    print(f"    {len(labels)} crops ({(labels == POSITIVE).sum()} positive, "
          f"{(labels == NEGATIVE).sum()} negative), {ignored} ignored"
          + (f", {from_gt} positives from GT boxes" if from_gt else ""))
    print(f"    {gt_count.sum()} GT boxes in "
          f"{(gt_count > 0).sum()} frames, {gt_missed.sum()} with no "
          f"proposal above IoU {args.pos_iou}")
    if not len(labels):
        raise SystemExit("no crops extracted")

    crops = {}
    for m in modalities:
        crops[m] = torch.from_numpy(np.stack(cuts[m]))
        cuts[m] = None                  # release the list before the next stack
    return {"crops": crops, "labels": torch.from_numpy(labels),
            "frames": np.array(frames, dtype=np.int64),
            "gt_count": gt_count, "gt_missed": gt_missed}


class CropDataset(Dataset):
    """Holds pre-extracted crops; flips are the only augmentation.

    Flips apply to every modality together, or the registration between them
    is destroyed. Crops are stored as one stacked tensor per modality.
    """

    def __init__(self, crops, labels, augment=False):
        self.crops = crops
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        crops = {m: v[index] for m, v in self.crops.items()}
        if self.augment:
            if torch.rand(1).item() < 0.5:
                crops = {m: torch.flip(v, [2]) for m, v in crops.items()}
            if torch.rand(1).item() < 0.5:
                crops = {m: torch.flip(v, [1]) for m, v in crops.items()}
        return crops, int(self.labels[index])


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


def predict(model, loader, device):
    """Classifier predictions for every eval crop, in dataset order."""
    model.eval()
    pred = []
    with torch.no_grad():
        for crops, _ in loader:
            crops = {m: v.to(device) for m, v in crops.items()}
            pred.extend(model(crops).argmax(dim=1).cpu().tolist())
    return np.array(pred, dtype=np.int64)


def score(pred, eval_set, image_min_positive):
    """Truth and prediction for both evaluations.

    Per leaf: every eval proposal, plus one false negative per GT box that
    no proposal matched. Per image: GT presence against the number of
    proposals classified positive.
    """
    missed = int(eval_set["gt_missed"].sum())
    leaf_truth = np.concatenate([eval_set["labels"].numpy(),
                                 np.full(missed, POSITIVE)])
    leaf_pred = np.concatenate([pred, np.full(missed, NEGATIVE)])

    n_frames = len(eval_set["gt_count"])
    hits = np.bincount(eval_set["frames"][pred == POSITIVE],
                       minlength=n_frames)
    image_truth = np.where(eval_set["gt_count"] > 0, POSITIVE, NEGATIVE)
    image_pred = np.where(hits >= image_min_positive, POSITIVE, NEGATIVE)
    return (leaf_truth, leaf_pred), (image_truth, image_pred)


def report(truth, pred, target_names, **kw):
    return classification_report(truth, pred, labels=[POSITIVE, NEGATIVE],
                                 target_names=target_names,
                                 zero_division=0, **kw)


def print_confusion(truth, pred, target_names):
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':<20}" + "".join(f"{n[:14]:>16}" for n in target_names))
    for name, row in zip(target_names,
                         confusion_matrix(truth, pred,
                                          labels=[POSITIVE, NEGATIVE])):
        print(f"{name:<20}" + "".join(f"{v:>16}" for v in row))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-samples", type=str, required=True,
                   help="dataset list file for training")
    p.add_argument("--eval-samples", type=str, required=True,
                   help="dataset list file for evaluation")
    p.add_argument("--calibration", type=str, required=True,
                   help="camera calibration json for the thermal warp")
    p.add_argument("--box-dir", type=str, default="combined_roboflow_sam3",
                   help="subdirectory under each root holding box pickles")
    p.add_argument("--positive-classes", type=str, nargs="+", required=True,
                   help="GT box class names forming the target disease")
    p.add_argument("--proposal-classes", type=str, nargs="+",
                   default=["healthy leaf", "unhealthy leaf"],
                   help="SAM3 box class names used as proposals; merged with "
                        "NMS across prompts before labelling")
    p.add_argument("--pos-iou", type=float, default=0.2,
                   help="a proposal with IoU above this against a target GT "
                        "box is positive, in training and evaluation; a GT "
                        "box with no proposal above it is a false negative")
    p.add_argument("--neg-iou", type=float, default=0.05,
                   help="training only: proposals at or below this IoU are "
                        "negative, those between this and --pos-iou are "
                        "ignored; set equal to --pos-iou for no ignore band")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU above which overlapping proposals are merged")
    p.add_argument("--gt-positives", type=int, default=0, choices=[0, 1],
                   help="1 to also train on crops of the GT boxes as extra "
                        "positives; evaluation stays on proposals only")
    p.add_argument("--image-min-positive", type=int, default=1,
                   help="positive proposals needed to call an image positive")
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
                   help="training only: cap on negative proposals kept per "
                        "frame; 0 keeps all. Evaluation keeps every proposal")
    p.add_argument("--size", type=int, default=224,
                   help="crop resolution fed to the towers")
    p.add_argument("--pad", type=float, default=0.1,
                   help="context margin around each box")
    p.add_argument("--frame-size", type=int, default=576,
                   help="letterboxed frame size from the loader")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for negative sampling and initialization")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker processes, for crop extraction "
                        "and for training")
    p.add_argument("--results", type=str, default="",
                   help="CSV file to append one row of metrics to")
    p.add_argument("--save", type=str, default="",
                   help="path to save the best checkpoint to")
    args = p.parse_args()

    if args.neg_iou > args.pos_iou:
        raise SystemExit("--neg-iou must not exceed --pos-iou")
    shared = set(args.positive_classes) & set(args.proposal_classes)
    if shared:
        raise SystemExit(f"classes both target and proposal: {sorted(shared)}")

    torch.manual_seed(args.seed)
    class_map = {name: GT for name in args.positive_classes}
    for name in args.proposal_classes:
        class_map[name] = PROPOSAL

    # common = dict(size=args.frame_size, use_depth="depth" in args.modalities,
    #               box_dir=args.box_dir)
    # train_data = MultiSpectralDetection(args.train_samples, args.calibration,
    #                                     class_map, **common)
    # eval_data = MultiSpectralDetection(args.eval_samples, args.calibration,
    #                                    class_map, **common)
    use_depth = "depth" in args.modalities
    train_data = MultiSpectralDetection(args.train_samples, args.calibration,
                                       class_map, size=args.frame_size,
                                       use_depth=use_depth,box_dir="combined_roboflow_sam3_with_tracking")
                                    #    use_depth=use_depth,box_dir="combined_roboflow_sam3")
    eval_data = MultiSpectralDetection(args.eval_samples, args.calibration,
                                      class_map, size=args.frame_size,
                                      use_depth=use_depth,box_dir="combined_roboflow_sam3")
    
    print(f"modalities: {'+'.join(args.modalities)}  "
          f"unfreeze: {args.unfreeze} from {args.unfreeze_from}")
    print(f"target: {', '.join(args.positive_classes)}")
    print(f"proposals: {', '.join(args.proposal_classes)}  "
          f"pos-iou {args.pos_iou}  neg-iou {args.neg_iou}  "
          f"nms-iou {args.nms_iou}  gt-positives {args.gt_positives}")

    print("extracting train proposals")
    train_set = build_set(train_data, args.modalities, args,
                          random.Random(args.seed), train=True)
    print("extracting eval proposals")
    eval_set = build_set(eval_data, args.modalities, args,
                         random.Random(args.seed + 1), train=False)

    train_loader = DataLoader(
        CropDataset(train_set["crops"], train_set["labels"], augment=True),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=args.workers)
    eval_loader = DataLoader(
        CropDataset(eval_set["crops"], eval_set["labels"], augment=False),
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
    counts = np.bincount(train_set["labels"].numpy(), minlength=2)
    if counts[POSITIVE] == 0:
        print("WARNING: no positive training proposals")
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    print(f"class weights: positive {weights[0]:.3f}, negative {weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    target_names = ["+".join(args.positive_classes), "negative"]
    pos_name = target_names[0]
    best_f1, best_epoch, best = -1.0, 0, None

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

        leaf, image = score(predict(model, eval_loader, device), eval_set,
                            args.image_min_positive)
        leaf_report = report(*leaf, target_names, output_dict=True)
        image_report = report(*image, target_names, output_dict=True)
        leaf_f1 = leaf_report[pos_name]["f1-score"]
        print(f"epoch {epoch + 1:>3}/{args.epochs}  "
              f"loss {running / len(train_loader):.4f}  "
              f"leaf F1 {leaf_f1:.4f}  "
              f"image F1 {image_report[pos_name]['f1-score']:.4f}")

        # Selection follows the per-leaf score, which is what training
        # optimizes; the image score is reported at the same epoch.
        if leaf_f1 > best_f1:
            best_f1, best_epoch = leaf_f1, epoch + 1
            best = (leaf, image, leaf_report, image_report)
            if args.save:
                torch.save({"epoch": epoch + 1, "leaf_f1": leaf_f1,
                            "modalities": args.modalities,
                            "positive_classes": args.positive_classes,
                            "proposal_classes": args.proposal_classes,
                            "pos_iou": args.pos_iou, "nms_iou": args.nms_iou,
                            "gt_positives": args.gt_positives,
                            "model_state_dict": model.state_dict()}, args.save)

    leaf, image, leaf_report, image_report = best
    print(f"\n--- best epoch {best_epoch}, leaf F1 {best_f1:.4f} ---")
    print(f"\nper leaf (includes {int(eval_set['gt_missed'].sum())} "
          f"unproposed GT boxes as false negatives)")
    print(report(*leaf, target_names, digits=4))
    print_confusion(*leaf, target_names)
    print(f"\nper image ({len(eval_set['gt_count'])} frames, positive when "
          f">= {args.image_min_positive} positive proposals)")
    print(report(*image, target_names, digits=4))
    print_confusion(*image, target_names)

    if args.results:
        row = {
            "positive_classes": "+".join(args.positive_classes),
            "proposal_classes": "+".join(args.proposal_classes),
            "modalities": "+".join(args.modalities),
            "unfreeze": args.unfreeze,
            "unfreeze_from": args.unfreeze_from,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "max_negatives": args.max_negatives,
            "pos_iou": args.pos_iou,
            "neg_iou": args.neg_iou,
            "nms_iou": args.nms_iou,
            "image_min_positive": args.image_min_positive,
            "gt_positives": args.gt_positives,
            "n_train": len(train_set["labels"]),
            "n_eval_proposals": len(eval_set["labels"]),
            "gt_missed": int(eval_set["gt_missed"].sum()),
        }
        for prefix, rep in (("leaf", leaf_report), ("image", image_report)):
            pos = rep[pos_name]
            row[f"{prefix}_support_pos"] = int(pos["support"])
            row[f"{prefix}_support_neg"] = int(rep["negative"]["support"])
            row[f"{prefix}_pos_precision"] = round(pos["precision"], 4)
            row[f"{prefix}_pos_recall"] = round(pos["recall"], 4)
            row[f"{prefix}_pos_f1"] = round(pos["f1-score"], 4)
        exists = os.path.exists(args.results)
        with open(args.results, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nappended to {args.results}")


if __name__ == "__main__":
    main()