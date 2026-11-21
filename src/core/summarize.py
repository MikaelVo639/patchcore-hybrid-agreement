"""Summarize paired benchmark results by category and bootstrap paired differences."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ('small_aupro_05', 'aupro_05', 'aupro_30', 'image_auroc')
CATEGORIES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper')


def paired_ci(values):
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(10_000, len(values))).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(run, seed_count, extra_runs=()):
    rows = [dict(row, run=directory.name) for directory in (run, *extra_runs)
            for file in sorted(directory.glob('*/*/metrics.json')) for row in json.loads(file.read_text())]
    results = {}
    for row in rows:
        key = row['method'], row['budget']
        pair = row['category'], row['seed']
        group = results.setdefault(key, {})
        if pair in group:
            raise ValueError(f'Duplicate pair {key} {pair}')
        group[pair] = row

    pairs = [(category, seed) for category in CATEGORIES for seed in range(seed_count)]
    for key, group in results.items():
        if set(group) != set(pairs):
            raise ValueError(f'Incomplete {key}: {len(group)}/{len(pairs)} pairs')
    if not results:
        raise ValueError('No result groups found')
    for pair in pairs:
        signatures = {(group[pair]['manifest_sha256'], group[pair]['feature_signature']) for group in results.values()}
        if len(signatures) != 1:
            raise ValueError(f'Mismatched data or features for {pair}')

    for budget in sorted({budget for _, budget in results if budget}):
        candidates = [group for (method, cap), group in results.items()
                      if cap == budget and method != 'fp32-budget']
        if candidates:
            results['normal-q95-selector', budget] = {
                pair: min((group[pair] for group in candidates),
                          key=lambda row: (row['calibration_error_q95'], row['artifact_bytes'], row['method']))
                for pair in pairs
            }

    totals = []
    for (method, budget), group in results.items():
        row = dict(method=method, budget=budget, n_pairs=len(group),
                   run=';'.join(sorted({item['run'] for item in group.values()})))
        for metric in (*METRICS, 'artifact_bytes', 'patches'):
            row[metric] = float(np.mean([item[metric] for item in group.values()]))
        if budget:
            baseline = results['fp32-budget', budget]
            larger = results['fp32-10pct', 0]
            comparators = [('cap', baseline), ('10pct', larger)]
            if method == 'uint4-full' and ('int4-full', budget) in results:
                comparators.append(('signed_int4', results['int4-full', budget]))
            for name, comparator in comparators:
                for metric in ('small_aupro_05', 'image_auroc'):
                    deltas = np.array([group[pair][metric] - comparator[pair][metric] for pair in pairs])
                    deltas = deltas.reshape(len(CATEGORIES), seed_count).mean(axis=1)
                    low, high = paired_ci(100 * deltas)
                    prefix = f'{metric}_delta_{name}_pp'
                    row.update({prefix: 100 * deltas.mean(), prefix + '_low': low, prefix + '_high': high})
        totals.append(row)
    totals.sort(key=lambda row: (row['budget'], -row['small_aupro_05']))
    selections = [dict(category=category, seed=seed, budget=budget, selected=row['method'], run=row['run'],
                       artifact=row['artifact'], calibration_error_q95=row['calibration_error_q95'])
                  for (method, budget), group in results.items() if method == 'normal-q95-selector'
                  for (category, seed), row in group.items()]
    return totals, selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--seed-count', type=int, required=True)
    parser.add_argument('--extra-runs', type=Path, nargs='*', default=())
    args = parser.parse_args()
    totals, selections = summarize(args.run, args.seed_count, args.extra_runs)
    columns = list(dict.fromkeys(key for row in totals for key in row))
    with (args.run / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(totals)
    if selections:
        with (args.run / 'selection.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selections[0]))
            writer.writeheader()
            writer.writerows(selections)
    for row in totals:
        print(f"{row['budget']:7} {row['method']:22} {100 * row['small_aupro_05']:6.2f}%"
              f" | Image AUROC {100 * row['image_auroc']:6.2f}% | {row['artifact_bytes'] / 1024:.1f} KiB")


if __name__ == '__main__':
    main()
