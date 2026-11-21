"""Optional, explicit Torchvision encoder; core/cache workflows never import Torch."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .cache import FeatureCache
from .datasets import load_manifest, resolve_dataset_path
from .io import environment, sha256, write_json


def extract_mvtec(
    manifest_path,
    root,
    output,
    backbone="wide_resnet50_2",
    weights="IMAGENET1K_V1",
    image_size=224,
    resize_size=256,
    batch_size=8,
    device="cpu",
):
    try:
        import torch
        import torch.nn.functional as functional
        import torchvision
        from PIL import Image
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transforms
    except ImportError as exc:
        raise RuntimeError('Image extraction requires: pip install -e ".[images]"') from exc
    if backbone not in ("resnet18", "wide_resnet50_2"):
        raise ValueError("Supported backbones: resnet18, wide_resnet50_2")
    if weights not in ("IMAGENET1K_V1", "none"):
        raise ValueError("Use explicit IMAGENET1K_V1 or none (software smoke tests only)")
    if min(image_size, resize_size, batch_size) < 1 or resize_size < image_size:
        raise ValueError("Positive sizes/batch size required, with resize_size >= image_size")
    document = load_manifest(manifest_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Feature output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if backbone == "resnet18":
        enum = torchvision.models.ResNet18_Weights
        builder = torchvision.models.resnet18
    else:
        enum = torchvision.models.Wide_ResNet50_2_Weights
        builder = torchvision.models.wide_resnet50_2
    model = builder(weights=None if weights == "none" else enum.IMAGENET1K_V1)
    encoder = create_feature_extractor(model, return_nodes={"layer2": "layer2", "layer3": "layer3"})
    encoder.to(device).eval().requires_grad_(False)
    signature_data = {
        "extractor_version": 1,
        "backbone": backbone,
        "weights": weights,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "resize_short_edge": resize_size,
        "center_crop": image_size,
        "normalization": "ImageNet mean/std",
        "layers": ["layer2", "layer3"],
        "local_pool": "avg3-stride1-pad1",
        "fusion": "bilinear-layer3-to-layer2-then-concat",
        "random_weight_seed": 0 if weights == "none" else None,
    }
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    metadata = {
        "dataset": document["dataset"],
        "category": document["category"],
        "synthetic": False,
        "feature_signature": signature,
        "extractor": signature_data,
        "manifest_sha256": sha256(manifest_path),
        "benchmark_eligible_weights": weights != "none",
    }
    started = perf_counter()
    for split in ("build", "calibration", "test"):
        records = [r for r in document["records"] if r["split"] == split]
        blocks, masks, labels = [], [], []
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            tensors = []
            for record in batch:
                with Image.open(resolve_dataset_path(root, record["path"])) as raw:
                    image = transforms.center_crop(
                        transforms.resize(
                            raw.convert("RGB"),
                            resize_size,
                            InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                        image_size,
                    )
                    tensor = transforms.to_tensor(image)
                    tensors.append(
                        transforms.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    )
                if split == "test":
                    labels.append(record["label"])
                    if record.get("mask"):
                        with Image.open(resolve_dataset_path(root, record["mask"])) as raw_mask:
                            mask = transforms.center_crop(
                                transforms.resize(
                                    raw_mask.convert("L"), resize_size, InterpolationMode.NEAREST
                                ),
                                image_size,
                            )
                            masks.append(np.asarray(mask) > 0)
                    else:
                        masks.append(np.zeros((image_size, image_size), dtype=bool))
            with torch.inference_mode():
                extracted = encoder(torch.stack(tensors).to(device))
                low = functional.avg_pool2d(extracted["layer2"], 3, stride=1, padding=1)
                high = functional.avg_pool2d(extracted["layer3"], 3, stride=1, padding=1)
                high = functional.interpolate(
                    high, size=low.shape[-2:], mode="bilinear", align_corners=False
                )
                fused = torch.cat((low, high), dim=1).permute(0, 2, 3, 1).contiguous()
                blocks.append(fused.cpu().numpy().astype(np.float32))
        FeatureCache(
            np.concatenate(blocks),
            np.array([r["image_id"] for r in records]),
            split,
            metadata,
            np.array(labels) if split == "test" else None,
            np.stack(masks) if split == "test" else None,
        ).save(output / f"{split}.npz")
    write_json(
        output / "extraction.json",
        {
            **metadata,
            "seconds": perf_counter() - started,
            "device": device,
            "batch_size": batch_size,
            "environment": environment(),
            "memory_note": "Caches are accumulated per split in RAM; not an out-of-core extractor",
        },
    )
    return output
