"""Train/evaluate a two-mask model on annotated 224px field-classifier crops.

Manifest CSV columns: image,leaf_mask,symptom_mask,label,group,split.
Paths are relative to the manifest. Use independent source-photo/plant groups
for train, val and test. Masks are binary PNGs aligned to the padded RGB crop.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from leaf_severity import condition_type, read_mask


class SeverityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, dilation=2, padding=2), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, dilation=4, padding=4), nn.ReLU())
        self.masks = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(16, 2, 1))

    def forward(self, images):
        detail = self.detail(images)
        context = F.interpolate(self.context(detail), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.masks(torch.cat((detail, context), dim=1))


class PairedMasks(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            rgb = np.array(image.convert("RGB"))
        leaf = read_mask(self.root / row["leaf_mask"])
        symptoms = read_mask(self.root / row["symptom_mask"])
        if rgb.shape != (224, 224, 3) or leaf.shape != (224, 224) or symptoms.shape != leaf.shape:
            raise ValueError(f"Expected aligned 224px crop and masks for {row['image']}")
        if leaf.sum() < 64 or (symptoms & ~leaf).any():
            raise ValueError(f"Invalid leaf or symptom outline for {row['image']}")
        if condition_type(row["label"]) == "healthy" and symptoms.any():
            raise ValueError(f"Healthy example has marked symptoms in {row['image']}")
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255, torch.from_numpy(
            np.stack((leaf, symptoms))).float(), row["label"]


def load_splits(path):
    splits, groups, images = defaultdict(list), {}, {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            group, split = row["group"].strip(), row["split"].strip()
            if not group or split not in ("train", "val", "test"):
                raise ValueError("Each row needs a source group and a train, val or test split.")
            image = (path.parent / row["image"]).resolve()
            if groups.setdefault(group, split) != split or images.setdefault(image, split) != split:
                raise ValueError(f"Image or source group crosses splits: {row['image']} / {group}")
            if condition_type(row["label"]) not in ("healthy", "disease_symptoms"):
                raise ValueError("This lesion model trains on healthy controls and diseases with area-based symptoms.")
            splits[split].append(row)
    return splits


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    rows = defaultdict(list)
    for rgb, target, labels in loader:
        predicted = model(rgb.to(device)).cpu() > 0
        truth = target.bool()
        leaf, symptoms = predicted[:, 0], predicted[:, 1]
        predicted_area = leaf.sum((1, 2))
        predicted_fraction = (leaf & symptoms).sum((1, 2)) / predicted_area.clamp_min(1)
        actual_fraction = truth[:, 1].sum((1, 2)) / truth[:, 0].sum((1, 2))
        error = 100 * (predicted_fraction - actual_fraction).abs()
        intersection = (predicted & truth).sum((2, 3))
        union = (predicted | truth).sum((2, 3))
        iou = torch.where(union > 0, intersection / union, float("nan"))
        for label, delta, pair, area in zip(labels, error, iou, predicted_area):
            rows[label].append((float(delta), *pair.tolist(), bool(area >= 64)))
    report = {}
    for label, values in rows.items():
        data = np.asarray(values)
        report[label] = {
            "n": len(data), "area_mae_percentage_points": float(data[:, 0].mean()),
            "leaf_iou": float(data[:, 1].mean()),
            "symptom_iou": float(np.nanmean(data[:, 2])) if np.isfinite(data[:, 2]).any() else None,
            "symptom_iou_n": int(np.isfinite(data[:, 2]).sum()),
            "usable_leaf_proportion": float(data[:, 3].mean()),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "train", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("Epochs and batch size must be positive.")
    if args.command != "inspect" and args.checkpoint is None:
        parser.error("--checkpoint is required for train and evaluate.")
    if args.command == "evaluate" and not args.checkpoint.is_file():
        parser.error(f"No severity checkpoint at {args.checkpoint}. Train with completed masks first.")
    splits = load_splits(args.manifest)
    required = {"inspect": ("train", "val", "test"), "train": ("train", "val"), "evaluate": ("test",)}
    first_missing = None
    empty = []
    for split in required[args.command]:
        rows = splits[split]
        if not rows:
            empty.append(split)
        complete = 0
        for row in rows:
            missing = [row[key] for key in ("image", "leaf_mask", "symptom_mask")
                       if not (args.manifest.parent / row[key]).is_file()]
            if missing:
                first_missing = first_missing or missing[0]
            else:
                complete += 1
        print(f"{split}: {complete}/{len(rows)} crops have an image and both masks.", flush=True)
    if args.command == "inspect":
        if first_missing:
            print(f"First missing file: {first_missing}")
        return
    if empty:
        parser.error(f"The manifest has no rows for: {', '.join(empty)}.")
    if first_missing:
        parser.error(f"Missing {first_missing}. Save the required mask pairs with annotate_leaf_severity.py first.")
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(args.device)
    model = SeverityNet().to(device)

    def loader(split):
        return DataLoader(PairedMasks(args.manifest.parent, splits[split]), args.batch, shuffle=split == "train")

    if args.command == "evaluate":
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state"])
        print(json.dumps(evaluate(model, loader("test"), device), indent=2))
        return
    if args.checkpoint.exists():
        parser.error("Choose a new checkpoint path for training.")
    train, validation = loader("train"), loader("val")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    train_labels = {row["label"] for row in splits["train"]}
    val_labels = {row["label"] for row in splits["val"]}
    if not val_labels <= train_labels:
        parser.error("Validation contains labels absent from training.")
    best = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for rgb, target, _ in train:
            target = target.to(device)
            logits = model(rgb.to(device))
            probabilities = logits.sigmoid()
            axes = (0, 2, 3)
            dice_loss = 1 - ((2 * (probabilities * target).sum(axes) + 1) /
                             ((probabilities + target).sum(axes) + 1)).mean()
            loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation, device)
        error = float(np.mean([m["area_mae_percentage_points"] for m in metrics.values()]))
        print(f"Epoch {epoch + 1}/{args.epochs} | validation macro area MAE {error:.2f} pp", flush=True)
        if error < best:
            best = error
            temporary = args.checkpoint.with_suffix(".tmp")
            torch.save({"state": model.state_dict(), "validation": metrics,
                        "classes": sorted(train_labels & val_labels)}, temporary)
            temporary.replace(args.checkpoint)


if __name__ == "__main__":
    main()
