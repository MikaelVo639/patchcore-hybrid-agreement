"""Fixed-grid feature caches, explicit split identity and cross-split ID checks."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureCache:
    features: np.ndarray
    image_ids: np.ndarray
    split: str
    metadata: dict
    labels: np.ndarray | None = None
    masks: np.ndarray | None = None

    def validate(self):
        self.features = np.asarray(self.features, dtype=np.float32)
        self.image_ids = np.asarray(self.image_ids, dtype=str)
        f = self.features
        if f.ndim != 4 or min(f.shape) == 0 or not np.isfinite(f).all():
            raise ValueError("Cache features must be finite [images, grid_h, grid_w, P]")
        if self.image_ids.shape != (len(f),) or len(set(self.image_ids)) != len(f):
            raise ValueError("Every image needs a unique ID")
        if self.split not in ("build", "calibration", "test"):
            raise ValueError("Unknown cache split")
        if self.split != "test" and (self.labels is not None or self.masks is not None):
            raise ValueError("Normal build/calibration caches must not contain labels or masks")
        if self.split == "test":
            if self.labels is None or self.masks is None:
                raise ValueError("Test cache requires labels and masks")
            self.labels, self.masks = np.asarray(self.labels), np.asarray(self.masks)
            if self.labels.shape != (len(f),) or not np.isin(self.labels, [0, 1]).all():
                raise ValueError("Test image labels must be binary [images]")
            if self.masks.ndim != 3 or self.masks.shape[0] != len(f) or min(self.masks.shape) == 0:
                raise ValueError("Test masks must be [images, height, width]")
            if not np.isin(self.masks, [0, 1]).all():
                raise ValueError("Test masks must be binary")
            if np.any(self.masks[self.labels == 0]):
                raise ValueError("Normal test images must have empty masks")
        if not isinstance(self.metadata, dict) or any(
            not isinstance(self.metadata.get(key), str) or not self.metadata[key]
            for key in ("feature_signature", "dataset", "category")
        ):
            raise ValueError(
                "Cache metadata requires nonempty feature_signature, dataset and category"
            )
        return self

    def rows(self):
        return self.features.reshape(-1, self.features.shape[-1])

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {
            "features": self.features,
            "image_ids": self.image_ids,
            "split": np.array(self.split),
            "metadata_json": np.array(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.split == "test":
            fields.update(labels=self.labels.astype(np.uint8), masks=self.masks.astype(np.uint8))
        # File handle prevents NumPy silently appending an extension.
        with path.open("wb") as stream:
            np.savez(stream, **fields)

    @classmethod
    def load(cls, path, expected_split):
        with np.load(path, allow_pickle=False) as data:
            cache = cls(
                data["features"],
                data["image_ids"],
                str(data["split"].item()),
                json.loads(str(data["metadata_json"].item())),
                data["labels"] if "labels" in data else None,
                data["masks"] if "masks" in data else None,
            ).validate()
        if cache.split != expected_split:
            raise ValueError(f"Expected {expected_split}, got {cache.split}")
        return cache


def check_compatible(*caches):
    seen = set()
    reference = caches[0]
    for cache in caches:
        cache.validate()
        if reference.metadata["feature_signature"] != cache.metadata["feature_signature"]:
            raise ValueError("Feature signatures differ: encoder/preprocessing/cache mismatch")
        if any(reference.metadata[key] != cache.metadata[key] for key in ("dataset", "category")):
            raise ValueError("Dataset/category identities differ across caches")
        if reference.features.shape[1:] != cache.features.shape[1:]:
            raise ValueError("Patch grid or feature dimensions differ")
        overlap = seen & set(cache.image_ids)
        if overlap:
            raise ValueError(f"Image IDs overlap across splits: {sorted(overlap)[:3]}")
        seen.update(cache.image_ids)
