"""Compare healthy-patch novelty with the supervised contrast masks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from healthy_patch_severity import OUTPUT as PATCH_OUTPUT, distances, patch_map
from leaf_severity import condition_type, overlay
from train_leaf_abnormality import binarize, masks_for, read_examples
from train_leaf_severity import load_splits
from train_severity_all_images import PILOT, SIZE, resize


CONTRAST = PILOT.parent / "healthy_contrast" / "abnormality.joblib"
DESTINATION = PILOT.parent / "hybrid_evaluation"


def predict_fields(rgb, label, contrast, patches):
    leaf_scores, supervised = masks_for(rgb, contrast, label)
    leaf, _ = binarize(leaf_scores, supervised, *contrast["thresholds"])
    reduced_leaf = cv2.resize(leaf.astype(np.uint8), (SIZE, SIZE),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    model = patches["models"].get(("plantdoc", label.partition("___")[0]))
    raw_novelty = np.zeros(leaf.shape, np.float32)
    if model is not None and (local := patch_map(resize(rgb), reduced_leaf)) is not None:
        features, core = local
        map_small = np.zeros(core.shape, np.float32)
        map_small[core] = distances(features[core], model) / max(model["threshold"], 1e-8)
        raw_novelty = cv2.resize(map_small, leaf.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return rgb, label, leaf, supervised, raw_novelty


def prepare_examples(manifest, split, contrast, patches):
    rows = load_splits(manifest)[split]
    return [(*predict_fields(rgb, label, contrast, patches), leaf_truth, symptom_truth)
            for rgb, (leaf_truth, symptom_truth), label in read_examples(manifest.parent, rows)]


def refine_leaf(example, radius):
    """Refine the leaf boundary while retaining a foreground core."""
    rgb, label, leaf, supervised, novelty, leaf_truth, symptom_truth = example
    if not radius or leaf.sum() < 64:
        return example

    pixels = leaf.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    core = cv2.erode(pixels, kernel).astype(bool)
    nearby = cv2.dilate(pixels, kernel).astype(bool)
    if core.sum() < 64 or (~nearby).sum() < 64:
        return example

    seeds = np.full(leaf.shape, cv2.GC_BGD, np.uint8)
    seeds[nearby] = cv2.GC_PR_BGD
    seeds[leaf] = cv2.GC_PR_FGD
    seeds[core] = cv2.GC_FGD
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros_like(background)
    cv2.setRNGSeed(42)
    cv2.grabCut(rgb, seeds, None, background, foreground, 2, cv2.GC_INIT_WITH_MASK)
    refined = (seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD)
    return rgb, label, refined, supervised, novelty, leaf_truth, symptom_truth


def symptom_mask(example, setting):
    _, _, leaf, supervised, novelty, _, _ = example
    weight, novelty_cutoff, score_cutoff = setting
    # A bounded novelty signal prevents a small number of extreme pixels dominating.
    novelty = novelty / (novelty + novelty_cutoff)
    combined = (1 - weight) * supervised + weight * novelty
    return leaf & (combined >= score_cutoff)


def metrics(rows, setting):
    area_errors, false_areas, symptom_ious, leaf_ious = [], [], [], []
    for example in rows:
        _, label, leaf, _, _, leaf_truth, symptom_truth = example
        mask = symptom_mask(example, setting)
        area = 100 * mask.sum() / max(1, leaf.sum())
        actual = 100 * symptom_truth.sum() / leaf_truth.sum()
        leaf_ious.append((leaf & leaf_truth).sum() / (leaf | leaf_truth).sum())
        if condition_type(label) == "healthy":
            false_areas.append(area)
        else:
            area_errors.append(abs(area - actual))
            symptom_ious.append((mask & symptom_truth).sum() / max(1, (mask | symptom_truth).sum()))
    mean = lambda values: float(np.mean(values)) if values else None
    return {"n": len(rows), "area_mae_pp": mean(area_errors), "symptom_iou": mean(symptom_ious),
            "healthy_false_area_pct": mean(false_areas), "leaf_iou": mean(leaf_ious)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PILOT)
    parser.add_argument("--test-manifest", type=Path, help="Evaluate the selected settings on a separate reviewed test set.")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--refine-leaf", action="store_true",
                        help="Select a GrabCut boundary refinement using validation leaf masks.")
    args = parser.parse_args()
    if args.refine_leaf and args.output.resolve() == DESTINATION.resolve():
        parser.error("Use a separate --output folder for boundary refinement.")
    cv2.setNumThreads(1)
    contrast = joblib.load(CONTRAST)
    patches = joblib.load(PATCH_OUTPUT / "healthy_patch.joblib")
    validation = prepare_examples(args.manifest, "val", contrast, patches)
    baseline = (0., 1., contrast["thresholds"][1])
    radius = 0
    if args.refine_leaf:
        boundary_candidates = {size: [refine_leaf(row, size) for row in validation] for size in (0, 4, 9)}
        boundary_validation = {size: metrics(rows, baseline) for size, rows in boundary_candidates.items()}
        radius = max(boundary_validation, key=lambda size: (boundary_validation[size]["leaf_iou"], -size))
        validation = boundary_candidates[radius]
        print(json.dumps({"leaf_refinement_radius": radius, "validation": boundary_validation}), flush=True)

    disease = defaultdict(list)
    for example in validation:
        if condition_type(example[1]) == "disease_symptoms":
            disease[example[1]].append(example)
    healthy = [row for row in validation if condition_type(row[1]) == "healthy"]
    if not disease or not healthy:
        parser.error("Validation needs both reviewed healthy and diseased images.")
    choices = [(weight, patch, confidence)
               for weight in (0., .2, .4, .6, .8)
               for patch in (.15, .3, .6, 1.)
               for confidence in (.45, .50, .55, .60, .65, .70, .75)]
    report = {}
    settings = {}
    for label, examples in disease.items():
        rows = examples + healthy
        original = metrics(rows, baseline)
        # Require healthy false area no higher than the deployed contrast baseline.
        admissible = []
        for params in choices:
            score = metrics(rows, params)
            if score["healthy_false_area_pct"] <= original["healthy_false_area_pct"]:
                quality = score["symptom_iou"] - score["area_mae_pp"] / 100
                admissible.append((quality, params))
        proposed = max(admissible, key=lambda row: row[0])[1] if admissible else baseline
        settings[label] = proposed
        report[label] = {"setting": proposed, "validation_baseline": original,
                         "validation_candidate": metrics(rows, proposed)}

    # Test crops are read after the settings are fixed.
    test_manifest = args.test_manifest or args.manifest
    if args.test_manifest:
        if args.output.resolve() == DESTINATION.resolve():
            parser.error("Use a separate --output folder for the new test results.")
        original_groups = {row["group"].removesuffix(".0") for rows in load_splits(args.manifest).values()
                           for row in rows}
        test_rows = load_splits(test_manifest)["test"]
        if not test_rows or any(row["group"].removesuffix(".0") in original_groups for row in test_rows):
            parser.error("The separate test set needs new source groups.")
        missing = sum(not all((test_manifest.parent / row[key]).is_file()
                              for key in ("image", "leaf_mask", "symptom_mask")) for row in test_rows)
        if missing:
            parser.error(f"Add reviewed masks for {missing} test crops before scoring.")
    test = prepare_examples(test_manifest, "test", contrast, patches)
    test_healthy = [row for row in test if condition_type(row[1]) == "healthy"]
    args.output.mkdir(parents=True, exist_ok=True)
    for label, params in settings.items():
        examples = [row for row in test if row[1] == label]
        rows = examples + test_healthy
        original = metrics(rows, baseline)
        refined = [refine_leaf(row, radius) for row in rows]
        candidate = metrics(refined, params)
        report[label].update(test_baseline=original, test_candidate=candidate)
        if args.refine_leaf:
            report[label]["leaf_refinement_radius"] = radius
        if examples:
            image, _, leaf, *_ = refined[0]
            overlay(image, leaf, symptom_mask(refined[0], params)).save(args.output / f"{label}_overlay.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
