"""Dependency-free Markdown and SVG summaries. No inference from synthetic results."""

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import write_csv, write_json

METRICS = ("image_auroc", "pixel_auroc", "aupro_0_3")


def _value(value):
    return "undefined" if value is None else f"{value:.4f}"


def _plot(rows, output):
    colors = ["#1e40af", "#b91c1c", "#15803d", "#7e22ce", "#a16207", "#0e7490", "#475569"]
    width, height = 800, 420
    budgets = [r["budget_bytes"] for r in rows]
    lo, hi = min(budgets), max(budgets)

    def x(b):
        return 70 + 650 * (b - lo) / max(hi - lo, 1)

    def y(score):
        return 335 - 260 * score

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial,sans-serif" font-size="13" fill="#172033">',
        '<text x="70" y="25" font-size="18">Image AUROC versus storage budget</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        items += [
            f'<path d="M70 {y(tick)} H720" stroke="#d8dce3"/>',
            f'<text x="25" y="{y(tick) + 4}">{tick:.2f}</text>',
        ]
    for budget in sorted(set(budgets)):
        items.append(
            f'<text x="{x(budget)}" y="357" text-anchor="middle">{budget / 1024:g} KiB</text>'
        )
    groups = defaultdict(list)
    for row in rows:
        if row["image_auroc"] is not None:
            groups[row["method"]].append(row)
    for i, (method, points) in enumerate(sorted(groups.items())):
        color = colors[i % len(colors)]
        points.sort(key=lambda row: row["budget_bytes"])
        coords = " ".join(f"{x(r['budget_bytes'])},{y(r['image_auroc'])}" for r in points)
        items.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in points:
            items.append(
                f'<circle cx="{x(row["budget_bytes"])}" cy="{y(row["image_auroc"])}" r="3" fill="{color}"/>'
            )
        items.append(
            f'<text x="{70 + (i % 4) * 180}" y="{385 + (i // 4) * 20}" fill="{color}">{html.escape(method)}</text>'
        )
    items.append("</g></svg>")
    Path(output).write_text("\n".join(items), encoding="utf-8")


def build_report(run_directories, output):
    rows, inputs = [], []
    for directory in run_directories:
        directory = Path(directory)
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError(f"Cannot report an incomplete run: {directory}")
        batch = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        rows.extend(batch)
        inputs.append(str(directory))
    if not rows:
        raise ValueError("No evaluated models to report")
    identities = [
        (r["dataset"], r["category"], r["method"], r["budget_bytes"], r["seed"]) for r in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate category/method/budget/seed results; do not count a run twice")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Report output must be empty; choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    by_category = defaultdict(list)
    for row in rows:
        by_category[(row["dataset"], row["category"], row["method"], row["budget_bytes"])].append(
            row
        )
    categories = []
    for key, group in sorted(by_category.items()):
        row = dict(zip(("dataset", "category", "method", "budget_bytes"), key, strict=True))
        row["seed_count"] = len(group)
        row["seeds"] = ",".join(str(g["seed"]) for g in sorted(group, key=lambda g: g["seed"]))
        for metric in METRICS:
            values = [g[metric] for g in group if g[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        categories.append(row)
    by_dataset = defaultdict(list)
    for row in categories:
        by_dataset[(row["dataset"], row["method"], row["budget_bytes"])].append(row)
    summary = []
    for key, group in sorted(by_dataset.items()):
        row = dict(zip(("dataset", "method", "budget_bytes"), key, strict=True))
        row["category_count"] = len(group)
        row["categories"] = ",".join(g["category"] for g in group)
        row["seed_sets_match"] = len({g["seeds"] for g in group}) == 1
        for metric in METRICS:
            values = [g[metric] for g in group if g[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_defined_categories"] = len(values)
        summary.append(row)
    write_csv(output / "category_summary.csv", categories)
    write_csv(output / "dataset_summary.csv", summary)
    write_json(
        output / "report_inputs.json",
        {
            "runs": inputs,
            "aggregation": "seed mean within category, then equal-category mean within dataset",
        },
    )
    synthetic = any(r.get("synthetic", False) for r in rows)
    lines = [
        "# Experiment report",
        "",
        "**SYNTHETIC SOFTWARE DEMO — not benchmark evidence.**"
        if synthetic
        else "Results from the supplied completed runs; inspect coverage and protocol before drawing conclusions.",
        "",
        "Aggregation: mean across seeds within each category, then equal-weight category mean per dataset.",
        "Missing/undefined metrics are excluded and their category counts are reported in CSV.",
        "Compare methods only when their category and seed coverage match. Seed standard deviations are descriptive, not confidence intervals.",
        "",
        "| Dataset | Method | Budget (bytes) | Categories | Image AUROC | Pixel AUROC | AUPRO @ 0.3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['budget_bytes']} | {row['category_count']} | "
            + " | ".join(_value(row[m]) for m in METRICS)
            + " |"
        )
    for dataset in sorted({r["dataset"] for r in summary}):
        filename = f"quality_memory_{len([x for x in lines if x.startswith('![')])}.svg"
        _plot([r for r in summary if r["dataset"] == dataset], output / filename)
        lines += ["", f"![{dataset} quality versus budget]({filename})"]
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The whitening variant is a local PCA-whitening ablation, not a reproduction of streaming MH-PatchCore.",
        "Feature-only latency excludes the encoder. Peak process memory is cumulative within each run.",
        "Test-best configurations must only be used for post-hoc oracle analysis; never retune the selector using test labels.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
