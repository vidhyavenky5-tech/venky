"""Train MobileNetV3 on both domains, selecting checkpoints on PlantDoc macro F1."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from train_plantdoc_mobilenet import (ALIGNED, HERE, OUT_ROOT, CropDataset,
                                      classification_metrics, evaluate, make_model, padded_box_crop)


COMBINED = HERE / "preprocessed" / "combined42"
BASELINE = OUT_ROOT / "field_mnv3_224_s42"
BACKBONE = "mobilenet_v3_large"


class ArrayImages(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1)
        if image.shape[-1] != 224:
            image = torch.nn.functional.interpolate(
                image[None].float(), size=(224, 224), mode="bilinear", align_corners=False
            )[0].round().to(torch.uint8)
        return self.transform(image), int(self.labels[index])


def load_inputs():
    classes = json.loads((COMBINED / "class_map.json").read_text())["classes"]
    if len(classes) != 42 or len(set(classes)) != 42:
        raise ValueError("Expected 42 distinct combined classes.")
    manifest_path = ALIGNED / "crop_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    crops = manifest[manifest.status.eq("crop")]
    if crops.parent_group.isna().any() or crops.groupby("parent_group").split.nunique().gt(1).any():
        raise ValueError("PlantDoc source groups must stay within one split.")
    if not set(crops.label) <= set(classes):
        raise ValueError(f"Unmapped PlantDoc labels: {set(crops.label) - set(classes)}")
    splits = {split: crops[crops.split.eq(split)].reset_index(drop=True)
              for split in ("train", "val", "test")}
    baseline_classes = json.loads((BASELINE / "class_map.json").read_text())["classes"]
    with np.load(BASELINE / "test_predictions.npz") as baseline:
        expected = splits["test"].label.map({name: i for i, name in enumerate(baseline_classes)})
        if not np.array_equal(expected.to_numpy(), baseline["truth"]):
            raise ValueError("PlantDoc test rows differ from the saved baseline.")
    signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
    return classes, splits, signature


def plantdoc_pixels(frame, split, signature):
    """Build each crop once on disk instead of decoding on every epoch."""
    directory = ALIGNED / "mobilenet_rgb_cache"
    directory.mkdir(exist_ok=True)
    path = directory / f"{split}_{signature}.npy"
    if not path.exists():
        import tempfile
        shape = (len(frame), 224, 224, 3)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            staging = Path(temporary) / "pixels.npy"
            pixels = np.lib.format.open_memmap(staging, mode="w+", dtype=np.uint8, shape=shape)
            for i, row in enumerate(tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Cache {split}")):
                pixels[i] = padded_box_crop(row.source, row.box).permute(1, 2, 0).numpy()
            pixels.flush()
            del pixels
            staging.replace(path)
    return np.load(path, mmap_mode="r")


def load_split(split, classes, frame, signature, augment=False):
    class_to_id = {name: i for i, name in enumerate(classes)}
    with np.load(COMBINED / f"existing_{split}.npz") as cache:
        existing = cache["X"], cache["y"]
    plantdoc = CropDataset(frame, class_to_id, train=augment)
    return {
        "existing": ArrayImages(*existing, plantdoc.transform),
        "plantdoc": ArrayImages(plantdoc_pixels(frame, split, signature), plantdoc.y, plantdoc.transform),
    }


def balanced_batches(existing_n, field_n, batch_size, generator):
    """Visit every existing index once and cycle shuffled PlantDoc indices."""
    half = batch_size // 2
    existing = torch.randperm(existing_n, generator=generator)
    needed = max(existing_n, field_n)
    field = torch.cat([torch.randperm(field_n, generator=generator)
                       for _ in range((needed + field_n - 1) // field_n)])[:needed]
    for start in range(0, needed, half):
        end = start + half
        yield existing[start:end].tolist() + (existing_n + field[start:end]).tolist()


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "train", "evaluate"))
    parser.add_argument("--name", default="combined_mobile42_s42")
    parser.add_argument("--mixed-epochs", type=int, default=12)
    parser.add_argument("--field-epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, help="Warm-start a new run with compatible weights.")
    parser.add_argument("--balance-field", action="store_true",
                        help="Mix natural sampling with inverse-square-root class balancing during field fine-tuning.")
    args = parser.parse_args()
    if args.batch < 2 or args.batch % 2 or args.mixed_epochs < 0 or args.field_epochs < 1:
        parser.error("Use an even batch size, a non-negative mixed phase, and at least one field epoch.")
    if args.command == "train" and args.mixed_epochs == 0 and not (args.init_checkpoint or args.resume):
        parser.error("Field-only training requires --init-checkpoint or --resume.")
    if args.resume and args.init_checkpoint:
        parser.error("Use --resume or --init-checkpoint for training.")
    classes, splits, signature = load_inputs()
    if args.command == "inspect":
        for split, frame in splits.items():
            with np.load(COMBINED / f"existing_{split}.npz") as cache:
                labels = cache["y"]
            print(f"{split}: existing {len(labels):,} across {len(set(labels))} classes | "
                  f"PlantDoc {len(frame):,} across {frame.label.nunique()} classes")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    output = OUT_ROOT / args.name
    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output / "best.pt", output / "last.pt"
    fresh = args.command == "train" and not (args.resume or args.init_checkpoint)
    model = make_model(BACKBONE, len(classes), pretrained=fresh).to(device)
    loader_args = {"num_workers": 0, "pin_memory": device.type == "cuda"}

    def score_split(datasets, split):
        metrics = {}
        test_predictions = []
        for source in ("plantdoc", "existing"):
            loader = DataLoader(datasets[source], batch_size=args.batch, **loader_args)
            result, table, truth, prediction = evaluate(model, loader, device, classes, f"{source} {split}")
            metrics[source] = {"n": len(truth), **result}
            if split == "test":
                test_predictions.append((truth, prediction))
                table.to_csv(output / f"{source}_test_per_class.csv", index=False)
                np.savez_compressed(output / f"{source}_test_predictions.npz", truth=truth, prediction=prediction)
        if test_predictions:
            truth = np.concatenate([labels for labels, _ in test_predictions])
            prediction = np.concatenate([labels for _, labels in test_predictions])
            overall, _ = classification_metrics(truth, prediction, classes)
            metrics["overall"] = {"n": len(truth), "correct": int((truth == prediction).sum()), **overall}
        return metrics

    if args.command == "evaluate":
        saved = torch.load(best_path, map_location=device, weights_only=True)
        if saved["classes"] != classes or saved["manifest_signature"] != signature:
            raise ValueError("Checkpoint classes or PlantDoc split changed.")
        model.load_state_dict(saved["state"])
        datasets = load_split("test", classes, splits["test"], signature)
        metrics = score_split(datasets, "test")
        baseline = json.loads((BASELINE / "test_results.json").read_text())
        original_classes = pd.read_csv(BASELINE / "test_per_class.csv").set_index("label")
        updated_classes = pd.read_csv(output / "plantdoc_test_per_class.csv")
        if not updated_classes.set_index("label").loc[original_classes.index, "n"].eq(original_classes["n"]).all():
            raise ValueError("PlantDoc per-class test support differs from the baseline.")
        supported = updated_classes[updated_classes.n.gt(0)].copy()
        supported["baseline_recall"] = supported.label.map(original_classes["recall"])
        supported["baseline_f1"] = supported.label.map(original_classes["f1"])
        supported["recall_change_pp"] = 100 * (supported.recall - supported.baseline_recall)
        supported["f1_change_pp"] = 100 * (supported.f1 - supported.baseline_f1)
        supported.sort_values("recall").to_csv(output / "plantdoc_per_class_comparison.csv", index=False)
        report = {"primary": "plantdoc", "selected_epoch": saved["epoch"], **metrics,
                  "plantdoc_baseline": baseline,
                  "plantdoc_accuracy_change_pp": 100 * (metrics["plantdoc"]["accuracy"] - baseline["accuracy"]),
                  "plantdoc_macro_f1_change_pp": 100 * (metrics["plantdoc"]["macro_f1"] - baseline["macro_f1"])}
        (output / "test_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if not args.resume and (best_path.exists() or last_path.exists()):
        parser.error("Run already exists. Use --resume or a different --name.")
    settings = {key: getattr(args, key) for key in ("mixed_epochs", "field_epochs", "batch")}
    settings["manifest_signature"] = signature
    if args.balance_field:
        settings["balance_field"] = True
    previous = torch.load(last_path, map_location=device, weights_only=True) if args.resume else None
    if previous and (previous["settings"] != settings or previous["classes"] != classes):
        parser.error("Resume requires the original split and run settings.")
    if previous:
        model.load_state_dict(previous["state"])
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial["classes"] != classes or initial["manifest_signature"] != signature or initial["config"]["backbone"] != BACKBONE:
            parser.error("Initialization requires a checkpoint from the same backbone, labels, and split.")
        model.load_state_dict(initial["state"])
    train = load_split("train", classes, splits["train"], signature, augment=True)
    validation = load_split("val", classes, splits["val"], signature)
    existing, field = train["existing"], train["plantdoc"]
    mixed = ConcatDataset((existing, field))
    if args.balance_field:
        counts = np.bincount(field.labels, minlength=len(classes))
        class_weights = counts[field.labels] ** -0.5
        field_weights = 0.5 / len(field) + 0.5 * class_weights / class_weights.sum()
    best = previous["best"] if previous else -1
    history = previous["history"] if previous else []
    start = previous["epoch"] + 1 if previous else 1
    total_epochs = args.mixed_epochs + args.field_epochs
    loss_function = nn.CrossEntropyLoss(label_smoothing=.05)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(start, total_epochs + 1):
        torch.manual_seed(42 + epoch)
        generator = torch.Generator().manual_seed(42 + epoch)
        field_only = epoch > args.mixed_epochs
        phase = "plantdoc_finetune" if field_only else "mixed"
        if epoch == start or epoch == args.mixed_epochs + 1:
            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": 1e-5 if field_only else 3e-5},
                {"params": model.classifier.parameters(), "lr": 5e-5 if field_only else 3e-4},
            ])
            if previous and epoch == start and previous["phase"] == phase:
                optimizer.load_state_dict(previous["optimizer"])
        if field_only and args.balance_field:
            sampler = WeightedRandomSampler(field_weights, len(field), generator=generator)
            loader = DataLoader(field, batch_size=args.batch, sampler=sampler, **loader_args)
        elif field_only:
            loader = DataLoader(field, batch_size=args.batch, shuffle=True, generator=generator, **loader_args)
        else:
            batches = list(balanced_batches(len(existing), len(field), args.batch, generator))
            loader = DataLoader(mixed, batch_sampler=batches, **loader_args)
        model.train()
        progress = tqdm(loader, desc=f"{phase} {epoch}/{total_epochs}")
        for images, targets in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = loss_function(model(images.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss. Last full epoch remains resumable.")
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")
        metrics = score_split(validation, "validation")
        row = {"epoch": epoch, "phase": phase, **metrics}
        history.append(row)
        score = metrics["plantdoc"]["macro_f1"]
        payload = {"state": model.state_dict(), "classes": classes, "epoch": epoch,
                   "config": {"backbone": BACKBONE}, "manifest_signature": signature}
        if score > best:
            best = score
            atomic_save({**payload, "validation": metrics}, best_path)
        atomic_save({**payload, "phase": phase, "settings": settings, "history": history,
                     "best": best, "optimizer": optimizer.state_dict()}, last_path)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
