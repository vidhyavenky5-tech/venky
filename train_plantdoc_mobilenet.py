"""Train a PlantDoc field-specialist on original-resolution annotated leaf crops.

The existing 42-class ANN remains the compact ARM/SNN candidate.  This script
is a separate, higher-capacity field benchmark: it learns only the classes
actually represented by PlantDoc development data, never reads the frozen test
split during training/model selection, and saves an ExecuTorch-ready PyTorch
checkpoint if its held-out test result justifies deployment work.

Default backbone: ImageNet-pretrained MobileNetV3-Large from TorchVision.
It has 5.48M parameters / 0.22 GFLOPs at 224px and is an Arm-feasible model
after INT8 export, but it is NOT compatible with the project's NumPy
``arm_infer.py`` runtime.  Do not activate it there.
"""
import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)
from torchvision.transforms import v2
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ALIGNED = HERE / "preprocessed" / "leaf_aligned42"
OUT_ROOT = ALIGNED / "field_model_runs"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def padded_box_crop(path, encoded_box=None, size=224, *, include_quality=False):
    """Crop an annotated leaf with margin, preserve tissue RGB and pad square.

    This is intentionally not a colour transfer or a synthetic leaf mask.  It
    is the deterministic inference contract for a detector-produced leaf box.
    """
    encoded_image = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR) if encoded_image.size else None
    if bgr is None:
        raise RuntimeError(f"Unreadable image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if encoded_box is None else map(int, json.loads(encoded_box))
    margin = round(0.08 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)
    leaf = rgb[y0:y1, x0:x1]
    if leaf.size == 0:
        raise RuntimeError(f"Invalid crop in {path}: {encoded_box}")
    scale = size / max(leaf.shape[:2])
    nh, nw = max(1, round(leaf.shape[0] * scale)), max(1, round(leaf.shape[1] * scale))
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    resized = cv2.resize(leaf, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1)
    if not include_quality:
        return tensor
    gray = cv2.cvtColor(leaf, cv2.COLOR_RGB2GRAY)
    # Standardize scale for the focus proxy. The gray padding is excluded.
    focus = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return tensor, {
        "short_side": min(gray.shape),
        "focus": float(cv2.Laplacian(focus, cv2.CV_32F).var()),
        "clipped": float(np.mean((gray <= 8) | (gray >= 247))),
    }


def crop_scores(model, images, mirror=False):
    scores = model(images).softmax(dim=1)
    if mirror:
        scores = (scores + model(images.flip(-1)).softmax(dim=1)) / 2
    return scores


class CropDataset(Dataset):
    def __init__(self, frame, class_to_id, train=False, size=224):
        self.frame = frame.reset_index(drop=True)
        self.y = self.frame.label.map(class_to_id).astype(np.int64).to_numpy()
        self.size = size
        self.transform = (
            v2.Compose([
                v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                v2.RandomAffine(degrees=25, translate=(0.08, 0.08), scale=(0.82, 1.18), interpolation=v2.InterpolationMode.BILINEAR),
                v2.ColorJitter(brightness=0.35, contrast=0.30, saturation=0.20, hue=0.04),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.12),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomErasing(p=0.12, scale=(0.01, 0.05), ratio=(0.4, 2.5), value="random"),
                v2.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]) if train else v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGE_MEAN, IMAGE_STD)])
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.transform(padded_box_crop(row.source, row.box, self.size)), int(self.y[index])


def make_model(backbone, num_classes, pretrained):
    if backbone == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    # TorchVision uses Dropout+Linear in both heads, but MobileNet has two
    # additional intermediate layers while EfficientNet does not.
    index = 3 if backbone == "mobilenet_v3_large" else 1
    model.classifier[index] = nn.Linear(model.classifier[index].in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device, classes, description):
    model.eval(); prediction = []; truth = []
    for image, label in tqdm(loader, desc=description, leave=False, ncols=100):
        prediction.extend(model(image.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        truth.extend(label.tolist())
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    metrics, table = classification_metrics(truth, prediction, classes)
    return metrics, table, truth, prediction


def classification_metrics(truth, prediction, classes):
    labels = np.arange(len(classes))
    precision, recall, f1, n = precision_recall_fscore_support(truth, prediction, labels=labels, zero_division=0)
    present = n > 0
    table = pd.DataFrame({"label": classes, "n": n, "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1[present].mean()),
        "represented_classes": int(present.sum()),
    }, table


def save_checkpoint(path, model, classes, args, epoch, score):
    torch.save({"state": model.state_dict(), "classes": classes, "config": vars(args), "epoch": epoch, "selection_macro_f1": score, "input_contract": {
        "crop": "detected leaf box + 8% margin, square pad 128, resize 224",
        "normalization": {"mean": IMAGE_MEAN, "std": IMAGE_STD},
        "note": "Model is a PlantDoc-supported-class field specialist; no 42-class ARM activation."}}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--name", default="mobilenetv3_plantdoc224_s42")
    parser.add_argument("--backbone", choices=("mobilenet_v3_large", "efficientnet_b0"), default="mobilenet_v3_large")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted train command with identical settings")
    parser.add_argument("--min-train-box-side", type=int, default=0,
                        help="Optional minimum native annotation width and height, in pixels, for training only")
    args = parser.parse_args()
    output = OUT_ROOT / args.name
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(ALIGNED / "crop_manifest.csv")
    manifest = manifest[manifest.status.eq("crop") & manifest.split.isin(["train", "val", "test"])].copy()
    # Classes derive only from development data.  The test split remains unread
    # in train mode, including for class-list construction.
    development = manifest[manifest.split.isin(["train", "val"])]
    classes = sorted(development.label.unique())
    class_to_id = {label: index for index, label in enumerate(classes)}
    for split in ("train", "val", "test"):
        unknown = sorted(set(manifest[manifest.split.eq(split)].label) - set(classes))
        if unknown:
            raise RuntimeError(f"{split} has labels absent from development: {unknown}")
    (output / "class_map.json").write_text(json.dumps({"classes": classes, "class_to_id": class_to_id}, indent=2))
    if args.command == "evaluate":
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        if checkpoint["classes"] != classes: raise RuntimeError("Class map mismatch")
        model = make_model(checkpoint["config"]["backbone"], len(classes), pretrained=False).to(device)
        model.load_state_dict(checkpoint["state"])
        test = CropDataset(manifest[manifest.split.eq("test")], class_to_id, size=224)
        test_loader = DataLoader(test, args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
        result, table, truth, prediction = evaluate(model, test_loader, device, classes, "PlantDoc frozen test")
        table.to_csv(output / "test_per_class.csv", index=False)
        np.savez_compressed(output / "test_predictions.npz", truth=truth, prediction=prediction)
        (output / "test_results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2)); return
    train_frame = manifest[manifest.split.eq("train")]
    val_frame = manifest[manifest.split.eq("val")]
    if args.min_train_box_side < 0:
        parser.error("--min-train-box-side must be nonnegative")
    if args.min_train_box_side:
        def long_enough(encoded_box):
            left, top, right, bottom = json.loads(encoded_box)
            return min(right - left, bottom - top) >= args.min_train_box_side

        kept = train_frame[train_frame.box.map(long_enough)]
        missing = set(train_frame.label) - set(kept.label)
        if missing:
            raise ValueError(f"Box filter would remove every training example for {sorted(missing)}")
        print(f"Training box filter kept {len(kept)}/{len(train_frame)} crops")
        train_frame = kept
    train = CropDataset(train_frame, class_to_id, train=True)
    val = CropDataset(val_frame, class_to_id)
    counts = Counter(train.y.tolist())
    sample_weights = torch.tensor([counts[int(label)] ** -0.5 for label in train.y], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    loader_args = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, args.batch, sampler=sampler, **loader_args)
    val_loader = DataLoader(val, args.batch, shuffle=False, **loader_args)
    if args.resume and not (output / "last.pt").exists():
        raise RuntimeError(f"No resumable checkpoint: {output / 'last.pt'}")
    model = make_model(args.backbone, len(classes), pretrained=not args.no_pretrained and not args.resume).to(device)
    head_parameters = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.lr_backbone},
        {"params": head_parameters, "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0; history = []; start_epoch = 1
    if args.resume:
        last = torch.load(output / "last.pt", map_location=device, weights_only=False)
        previous = last["config"]
        if previous.get("min_train_box_side", 0) != args.min_train_box_side:
            raise RuntimeError("Resume requires unchanged --min-train-box-side")
        for key in ("name", "backbone", "epochs", "batch", "lr_head", "lr_backbone", "seed"):
            if previous[key] != getattr(args, key):
                raise RuntimeError(f"Resume requires unchanged --{key.replace('_', '-')}")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        best, history, start_epoch = last["best"], last["history"], last["epoch"] + 1
        torch.set_rng_state(last["torch_rng"].cpu()); np.random.set_state(last["numpy_rng"]); random.setstate(last["python_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all([state.cpu() for state in last["cuda_rng"]])
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time(); model.train(); losses = []
        for image, target in tqdm(train_loader, desc=f"{args.name} epoch {epoch}/{args.epochs}", ncols=100):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(image.to(device, non_blocking=True)), target.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(loss.item())
        result, _, _, _ = evaluate(model, val_loader, device, classes, "PlantDoc validation")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **result, "seconds": round(time.time() - started, 1)}
        history.append(row); pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if result["macro_f1"] > best:
            best = result["macro_f1"]
            save_checkpoint(output / "best.pt", model, classes, args, epoch, best)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
                    "config": vars(args), "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, output / "last.pt")
        scheduler.step()
        print(f"epoch {epoch}: val accuracy={result['accuracy']:.4f}, val macro-F1={result['macro_f1']:.4f}, best={best:.4f}, {row['seconds']:.0f}s", flush=True)
    (output / "run_contract.json").write_text(json.dumps({"architecture": args.backbone + " ImageNet pretrained", "classes": len(classes), "preprocessing": "original-resolution annotation crop; 8% context margin; square pad; resize 224", "selection": "PlantDoc validation macro-F1", "test": "frozen and only read by evaluate command", "deployment": "Export to ExecuTorch/INT8 only after frozen test review; not NumPy arm_infer compatible"}, indent=2))
    print(f"Training complete. Evaluate once with: python {Path(__file__).name} evaluate --name {args.name}")


if __name__ == "__main__":
    main()
