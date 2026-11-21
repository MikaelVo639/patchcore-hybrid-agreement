"""Run the packaged Hybrid Agreement model on an image."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in ('src', 'src/core', 'src/comparison', 'src/advanced'):
    sys.path.insert(0, str(ROOT / folder))

import numpy as np
import torch

from advanced import Predictor, encoders, predict_path
from experiment import start_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Output heatmap .npy file')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    bank = ROOT / 'models/banks' / f'{args.category}.zip'
    if bank.relative_to(ROOT).as_posix() not in manifest['files']:
        parser.error('Unknown category. Choose a bank name from models/banks.')
    if not args.image.is_file():
        parser.error(f'Image does not exist: {args.image}')
    if args.output.suffix != '.npy':
        parser.error('--output must end in .npy')
    if not torch.cuda.is_available():
        parser.error('This deployment uses FP16 encoders and requires CUDA.')
    config = json.loads((ROOT / 'config.json').read_text())
    if config['method'] != manifest['method']:
        raise ValueError('Config and checkpoint method differ')
    info = manifest['files'][bank.relative_to(ROOT).as_posix()]
    payload = bank.read_bytes()
    if len(payload) != info['bytes'] or hashlib.sha256(payload).hexdigest() != info['sha256']:
        raise ValueError('Bank checksum differs from models/manifest.json')
    with torch.inference_mode():
        device = start_device()
        model = Predictor(payload, config['method'], device)
        wrn, dino = encoders(config['method'], device, ROOT / 'models/encoders')
        score, maps = predict_path(args.image, model, wrn, dino, device)
    heatmap = maps[0]
    if heatmap.shape != (224, 224) or not np.isfinite(heatmap).all() or not np.isfinite(score).all():
        raise ValueError('Nonfinite prediction or unexpected heatmap shape')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, heatmap)
    print(json.dumps(dict(category=args.category, score=float(score[0]),
                          shape=list(heatmap.shape), heatmap=str(args.output))))


if __name__ == '__main__':
    main()
