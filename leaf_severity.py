"""Visible symptom burden for one leaf, with a reviewed-mask command-line mode."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 0


def condition_type(label):
    name = label.casefold()
    if "___" not in name or "background_without_leaves" in name:
        return "unsupported"
    if any(word in name for word in ("mite", "aphid", "whitefly", "thrips", "leafminer")):
        return "pest_damage"
    if any(word in name for word in ("curl", "mosaic", "greening", "virus")):
        return "diffuse_or_deformation"
    return "healthy" if name.endswith("___healthy") else "disease_symptoms"


def measure_severity(leaf, symptoms, label, evidence):
    """Measure projected visible area and flag masks that need review."""
    kind = condition_type(label)
    result = {"condition": label, "type": kind, "evidence": evidence,
              "status": "review", "visible_affected_pct": None, "area_band": None}
    leaf, symptoms = np.asarray(leaf, bool), np.asarray(symptoms, bool)
    if leaf.ndim != 2 or symptoms.shape != leaf.shape:
        raise ValueError("Leaf and symptom masks must share the same 2D pixel grid.")
    if kind == "unsupported":
        return dict(result, reason="A supported leaf condition is needed.")
    leaf_area = int(leaf.sum())
    if leaf_area < 64:
        return dict(result, reason="Leaf mask too small. Retake with one leaf filling the frame.")
    area = int((leaf & symptoms).sum())
    outside = int(symptoms.sum()) - area
    if outside > 0.05 * max(1, area + outside):
        return dict(result, reason="Symptom mask extends outside the leaf. Review both outlines.")
    fraction = area / leaf_area
    result.update(visible_affected_pct=round(100 * fraction, 2), leaf_pixels=leaf_area,
                  affected_pixels=area)
    if kind == "diffuse_or_deformation":
        return dict(result, reason="Record visible area and review curl, mosaic or deformation separately.")
    if kind == "pest_damage":
        return dict(result, status="visible_damage_measured",
                    reason="Use scouting to assess active pest counts or infestation.")
    if kind == "healthy" and area or kind == "disease_symptoms" and not area:
        return dict(result, reason="Classifier label and visible symptom mask disagree.")
    # Engineering display bands. Disease-specific scales require validation.
    band = "none_visible" if not area else "low" if fraction < .10 else "moderate" if fraction < .25 else "high"
    return dict(result, status="measured", area_band=band)


def overlay(rgb, leaf, symptoms):
    pixels = np.asarray(rgb).copy()
    pixels[~leaf] = (pixels[~leaf] * .45).astype(np.uint8)
    affected = leaf & symptoms
    pixels[affected] = (.45 * pixels[affected] + .55 * np.array([255, 55, 45])).astype(np.uint8)
    return Image.fromarray(pixels)


def demo_masks(image_path, annotation, label):
    """Load the weak labels for the exact, manually outlined demo photo."""
    if not annotation.is_file():
        return None
    recipe = json.loads(annotation.read_text(encoding="utf-8"))
    if label != recipe["condition"] or hashlib.sha256(Path(image_path).read_bytes()).hexdigest() != recipe["sha256"]:
        return None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    region = Image.new("1", image.size)
    ImageDraw.Draw(region).polygon([tuple(point) for point in recipe["region"]], fill=1)
    leaf = np.asarray(region, dtype=bool)
    hsv = np.asarray(image.convert("HSV"))
    patches = leaf & (hsv[..., 2] >= recipe["white_patch_min_value"]) & (
        hsv[..., 1] <= recipe["white_patch_max_saturation"])
    return image, leaf, patches


def sample_demo(image_path, annotation, label):
    masks = demo_masks(image_path, annotation, label)
    if masks is None:
        return None
    image, leaf, patches = masks
    percent = 100 * int(patches.sum()) / int(leaf.sum())
    return percent, overlay(image, leaf, patches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--leaf-mask", type=Path, required=True)
    parser.add_argument("--symptom-mask", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    leaf, symptoms = read_mask(args.leaf_mask), read_mask(args.symptom_mask)
    if rgb.shape[:2] != leaf.shape:
        parser.error("The image and masks must match pixel-for-pixel.")
    result = measure_severity(leaf, symptoms, args.label, "user_supplied_masks")
    args.output.mkdir(parents=True, exist_ok=True)
    overlay(rgb, leaf, symptoms).save(args.output / "severity_overlay.png")
    (args.output / "severity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
