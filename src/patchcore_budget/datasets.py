"""MVTec AD manifests: split normal training by image, keep test untouched."""

import json
from pathlib import Path

import numpy as np

from .io import write_json

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def prepare_mvtec(root, category, output, calibration_fraction=0.2, seed=0):
    root, output = Path(root).resolve(), Path(output)
    if not category or Path(category).name != category or category in (".", ".."):
        raise ValueError("Category must be one directory name")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0,1)")
    if output.exists():
        raise ValueError("Manifest already exists; choose a new output path")
    folder = root / category
    train = sorted(
        p for p in (folder / "train" / "good").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    test = sorted(p for p in (folder / "test").glob("*/*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(train) < 2 or not test:
        raise ValueError("Expected at least two train/good images and a nonempty test directory")
    count = min(len(train) - 1, max(1, int(round(len(train) * calibration_fraction))))
    calibration_ids = set(np.random.default_rng(seed).permutation(len(train))[:count].tolist())
    records = []
    for i, path in enumerate(train):
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "image_id": relative,
                "path": relative,
                "split": "calibration" if i in calibration_ids else "build",
            }
        )
    for path in test:
        defect = path.parent.name != "good"
        mask = folder / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
        if defect and not mask.is_file():
            raise ValueError(f"Missing anomaly mask: {mask}")
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "image_id": relative,
                "path": relative,
                "split": "test",
                "label": int(defect),
                "mask": mask.relative_to(root).as_posix() if defect else None,
            }
        )
    document = {
        "schema_version": 1,
        "dataset": "mvtec_ad",
        "category": category,
        "split_seed": seed,
        "calibration_fraction": calibration_fraction,
        "records": records,
    }
    validate_manifest(document)
    write_json(output, document)
    return document


def validate_manifest(document):
    if document.get("schema_version") != 1 or not document.get("records"):
        raise ValueError("Invalid or empty manifest")
    ids, paths, splits = set(), set(), set()
    for row in document["records"]:
        if row["image_id"] in ids or row["path"] in paths:
            raise ValueError("Duplicate image ID or path in manifest")
        ids.add(row["image_id"])
        paths.add(row["path"])
        split = row["split"]
        if split not in ("build", "calibration", "test"):
            raise ValueError("Invalid manifest split")
        splits.add(split)
        if split != "test" and ("label" in row or "mask" in row):
            raise ValueError("Training records cannot contain test labels/masks")
        if split == "test" and row.get("label") not in (0, 1):
            raise ValueError("Test records require binary labels")
    if splits != {"build", "calibration", "test"}:
        raise ValueError("Manifest must contain all three splits")
    return document


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def resolve_dataset_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Manifest path escapes the dataset root")
    if not candidate.is_file():
        raise ValueError(f"Dataset file does not exist: {candidate}")
    return candidate
