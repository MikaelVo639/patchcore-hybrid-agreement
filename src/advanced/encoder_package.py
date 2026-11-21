"""Export inference-ready encoder weights and verify exact model state."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torchvision.models import wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

from deployment import create_encoder
from experiment import array_save, start_device
from patchcore_budget.io import sha256
from raw_scores import RESULTS, make_dino


FILES = {'wrn': 'wide-resnet50-layer1-layer2-fp16.safetensors',
         'dino': 'dinov2-small-fp16.safetensors'}


def construct(kind, file, device):
    if kind == 'wrn':
        model = create_feature_extractor(wide_resnet50_2(weights=None),
                                         return_nodes={'layer1': 'layer1', 'layer2': 'layer2'})
    else:
        import timm

        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, dynamic_img_size=True)
    model.load_state_dict(load_file(str(file), device=str(device)), strict=True, assign=True)
    model.eval().requires_grad_(False)
    if kind == 'wrn':
        return model.to(device, memory_format=torch.channels_last)
    return model.to(device)


def export(directory):
    device = start_device()
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind, filename in FILES.items():
        original = create_encoder(device, 'layer1-layer2', True) if kind == 'wrn' else make_dino(device)
        path = directory / filename
        destination = path.with_suffix('.tmp')
        save_file({key: t.cpu().contiguous() for key, t in original.state_dict().items()}, str(destination))
        destination.replace(path)
        restored = construct(kind, path, device)
        if any(not torch.equal(t, restored.state_dict()[key]) for key, t in original.state_dict().items()):
            raise ValueError(f'Export changed the {kind} state')
        result[kind] = dict(file=filename, bytes=path.stat().st_size, sha256=sha256(path), exact_state=True)
        del original, restored
    array_save(directory / 'encoders.json', json.dumps(result, indent=2).encode())
    print(json.dumps(result, indent=2))


def load_packaged(directory, need_wrn, need_dino, device):
    manifest = json.loads((directory / 'encoders.json').read_text())
    result = []
    for kind, required in (('wrn', need_wrn), ('dino', need_dino)):
        if not required:
            result.append(None)
            continue
        path = directory / FILES[kind]
        if sha256(path) != manifest[kind]['sha256']:
            raise ValueError(f'Packaged {kind} checkpoint checksum mismatch')
        result.append(construct(kind, path, device))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS / 'encoders')
    with torch.inference_mode():
        export(parser.parse_args().output)


if __name__ == '__main__':
    main()
