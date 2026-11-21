"""Post-hoc pilot analysis, isolated from the normal-only selector."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import sha256, write_csv, write_json
from .selection import choose

METRICS = ("aupro_0_3", "image_auroc", "pixel_auroc")


def paired_differences(rows, method="joint", reference="fixed"):
    """Require exactly matching category/seed/budget coverage; never pool pixels."""
    groups = {method: {}, reference: {}}
    for row in rows:
        if row["method"] not in groups:
            continue
        key = tuple(row[k] for k in ("dataset", "category", "seed", "budget_bytes"))
        group = groups[row["method"]]
        if key in group:
            raise ValueError("Duplicate category/seed/budget/method result")
        group[key] = row
    if not groups[method] or groups[method].keys() != groups[reference].keys():
        raise ValueError("Paired analysis requires matching nonempty coverage")
    result = []
    for key, row in sorted(groups[method].items()):
        other = groups[reference][key]
        entry = {k: row[k] for k in ("dataset", "category", "seed", "budget_bytes")}
        entry.update(method=method, reference=reference)
        for metric in METRICS:
            entry[metric + "_delta"] = (
                None
                if row[metric] is None or other[metric] is None
                else row[metric] - other[metric]
            )
        result.append(entry)
    return result


def oracle_gaps(selections, candidates, include_raw=False):
    """Metric-specific test oracles within each selector's original geometry/grid."""
    result = []
    for selected in selections:
        method = selected["method"]
        if method not in ("joint", "reconstruction", "coverage", "whitened_joint"):
            continue
        variant = "whitened" if method == "whitened_joint" else "pca"
        eligible = [
            row
            for row in candidates
            if row["selector_candidate"]
            and (
                row["variant"] == variant
                or (include_raw and variant == "pca" and row["variant"] == "raw")
            )
            and all(row[k] == selected[k] for k in ("dataset", "category", "seed", "budget_bytes"))
        ]
        if selected["candidate_key"] not in {row["candidate_key"] for row in eligible}:
            raise ValueError("Selected candidate missing from its oracle grid")
        for metric in METRICS:
            defined = [row for row in eligible if row[metric] is not None]
            best = (
                min(defined, key=lambda r: (-r[metric], r["artifact_bytes"], r["dimension"]))
                if defined
                else None
            )
            gap = (
                None
                if best is None or selected[metric] is None
                else best[metric] - selected[metric]
            )
            if gap is not None and gap < -1e-10:
                raise ValueError("Oracle metric disagrees with selected-model metric")
            result.append(
                {
                    **{
                        k: selected[k]
                        for k in ("dataset", "category", "seed", "budget_bytes", "method")
                    },
                    "metric": metric,
                    "selected_dimension": selected["dimension"],
                    "oracle_dimension": None if best is None else best["dimension"],
                    "selected_value": selected[metric],
                    "oracle_value": None if best is None else best[metric],
                    "regret": gap,
                    "eligible_candidates": len(eligible),
                    "scope": "post-hoc test oracle; not a deployable selection rule",
                }
            )
    return result


def alpha_sensitivity(
    selections, candidates, weights=(0, 0.25, 0.5, 0.75, 1), tie_tolerance=0.0, include_raw=False
):
    """Exploratory counterfactual choices from normal losses, never test scores."""
    if not weights or any(not np.isfinite(w) or not 0 <= w <= 1 for w in weights):
        raise ValueError("Sensitivity weights must be finite and in [0,1]")
    result = []
    for selected in selections:
        if selected["method"] != "joint":
            continue
        eligible = [
            r
            for r in candidates
            if r["selector_candidate"]
            and (r["variant"] == "pca" or (include_raw and r["variant"] == "raw"))
            and all(r[k] == selected[k] for k in ("dataset", "category", "seed", "budget_bytes"))
        ]
        if not eligible:
            raise ValueError("No PCA-grid candidates for sensitivity analysis")
        for weight in weights:
            weighted = [
                {
                    **r,
                    "counterfactual_loss": weight * r["reconstruction"]
                    + (1 - weight) * r["coverage"],
                }
                for r in eligible
            ]
            best, _ = choose(weighted, "counterfactual_loss", tie_tolerance)
            result.append(
                {
                    **{
                        k: best[k]
                        for k in (
                            "dataset",
                            "category",
                            "seed",
                            "budget_bytes",
                            "candidate_key",
                            "dimension",
                            "bank_count",
                            *METRICS,
                        )
                    },
                    "reconstruction_weight": weight,
                    "scope": "exploratory counterfactual; no change to deployed selector",
                }
            )
    return result


def build_study_report(run_directories, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output must be new or empty")
    rows, candidates, inputs, configs = [], [], [], []
    split_identities, signatures, code_versions = {}, set(), set()
    for directory in map(Path, run_directories):
        if json.loads((directory / "status.json").read_text())["status"] != "complete":
            raise ValueError("Incomplete run in study analysis")
        provenance = json.loads((directory / "provenance.json").read_text())
        if not provenance.get("test_opened_after_selection"):
            raise ValueError("Missing selection-before-test provenance")
        for filename, field in (
            ("selected_configs.json", "selection_sha256_before_test"),
            ("candidate_manifest.json", "candidate_manifest_sha256_before_test"),
        ):
            if sha256(directory / filename) != provenance.get(field):
                raise ValueError("Persisted pre-test selection/candidate manifest was changed")
        for filename, field in (
            ("config.json", "config_sha256_before_test"),
            ("candidates.csv", "candidates_sha256_before_test"),
        ):
            if field in provenance and sha256(directory / filename) != provenance[field]:
                raise ValueError("Persisted pre-test configuration/candidate record was changed")
        metadata = provenance["metadata"]
        category_key = (metadata["dataset"], metadata["category"])
        split_identity = tuple(
            provenance[k] for k in ("build_sha256", "calibration_sha256", "test_sha256")
        )
        if category_key in split_identities and split_identities[category_key] != split_identity:
            raise ValueError("Algorithm-seed analysis cannot mix image splits/caches")
        split_identities[category_key] = split_identity
        signatures.add(metadata["feature_signature"])
        code_versions.add(json.loads((directory / "environment.json").read_text())["git_commit"])
        rows.extend(json.loads((directory / "metrics.json").read_text()))
        candidates.extend(json.loads((directory / "candidate_metrics.json").read_text()))
        config = json.loads((directory / "config.json").read_text())
        configs.append({k: v for k, v in config.items() if k != "seed"})
        inputs.append(
            {
                "run": str(directory.resolve()),
                "metrics_sha256": sha256(directory / "metrics.json"),
                "candidate_metrics_sha256": sha256(directory / "candidate_metrics.json"),
            }
        )
    if (
        not rows
        or any(config != configs[0] for config in configs)
        or len(signatures) != 1
        or len(code_versions) != 1
    ):
        raise ValueError("Study requires results with a common configuration apart from seed")
    identities = [
        tuple(r[k] for k in ("dataset", "category", "seed", "budget_bytes", "method")) for r in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate run identities; do not count a run twice")
    pairs = []
    for reference in sorted({r["method"] for r in rows} - {"joint"}):
        pairs.extend(paired_differences(rows, reference=reference))
    include_raw = configs[0].get("include_raw_in_joint", False)
    gaps = oracle_gaps(rows, candidates, include_raw=include_raw)
    sensitivity = alpha_sensitivity(
        rows,
        candidates,
        tie_tolerance=configs[0].get("tie_tolerance", 0.0),
        include_raw=include_raw,
    )
    alpha_macro = []
    for dataset, budget, weight in sorted(
        {(r["dataset"], r["budget_bytes"], r["reconstruction_weight"]) for r in sensitivity}
    ):
        group = [
            r
            for r in sensitivity
            if (r["dataset"], r["budget_bytes"], r["reconstruction_weight"])
            == (dataset, budget, weight)
        ]
        categories = sorted({r["category"] for r in group})
        entry = dict(
            dataset=dataset,
            budget_bytes=budget,
            reconstruction_weight=weight,
            category_count=len(categories),
        )
        for metric in METRICS:
            category_means = []
            for category in categories:
                values = [
                    r[metric] for r in group if r["category"] == category and r[metric] is not None
                ]
                if values:
                    category_means.append(float(np.mean(values)))
            entry[metric] = float(np.mean(category_means)) if category_means else None
            entry[metric + "_defined_categories"] = len(category_means)
        alpha_macro.append(entry)
    category_pairs = defaultdict(list)
    for row in pairs:
        category_pairs[
            (row["dataset"], row["category"], row["budget_bytes"], row["reference"])
        ].append(row)
    category_summary = []
    for (dataset, category, budget, reference), group in sorted(category_pairs.items()):
        entry = dict(
            dataset=dataset,
            category=category,
            budget_bytes=budget,
            reference=reference,
            seed_count=len(group),
            seeds=",".join(str(r["seed"]) for r in group),
        )
        for metric in METRICS:
            values = [r[metric + "_delta"] for r in group if r[metric + "_delta"] is not None]
            entry[metric + "_delta"] = float(np.mean(values)) if values else None
            entry[metric + "_delta_seed_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
            entry[metric + "_defined_pairs"] = len(values)
        category_summary.append(entry)
    # Every category in a macro contrast must contain the same algorithm seeds.
    macro_groups = defaultdict(list)
    for row in category_summary:
        macro_groups[(row["dataset"], row["budget_bytes"], row["reference"])].append(row)
    macro = []
    for (dataset, budget, reference), group in sorted(macro_groups.items()):
        if len({r["seeds"] for r in group}) != 1:
            raise ValueError("Category seed coverage differs within a macro contrast")
        entry = dict(
            dataset=dataset, budget_bytes=budget, reference=reference, category_count=len(group)
        )
        for metric in METRICS:
            values = [r[metric + "_delta"] for r in group if r[metric + "_delta"] is not None]
            entry[metric + "_delta"] = float(np.mean(values)) if values else None
            entry[metric + "_min_category_delta"] = min(values) if values else None
            entry[metric + "_max_category_delta"] = max(values) if values else None
            entry[metric + "_defined_categories"] = len(values)
        macro.append(entry)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("paired_differences", pairs),
        ("paired_category_summary", category_summary),
        ("paired_macro_summary", macro),
        ("oracle_gaps", gaps),
        ("candidate_metrics", candidates),
        ("alpha_sensitivity", sensitivity),
        ("alpha_macro_summary", alpha_macro),
    ):
        write_json(output / f"{name}.json", data)
        write_csv(output / f"{name}.csv", data)
    write_json(
        output / "analysis_inputs.json",
        {
            "analysis_module_sha256": sha256(Path(__file__)),
            "run_code_versions": sorted(str(value) for value in code_versions),
            "feature_signatures": sorted(signatures),
            "inputs": inputs,
            "configuration": configs[0],
            "scope": "exploratory pilot; no population CI or significance test",
        },
    )

    def fmt(value):
        return "undefined" if value is None else f"{100 * value:+.3f}"

    lines = [
        "# Allocation pilot analysis",
        "",
        "**Exploratory pilot; not a final benchmark.**",
        "",
        "Differences are joint minus reference, in percentage points. Average seeds within category,",
        "then give categories equal weight. Category ranges are descriptive, not confidence intervals.",
        "",
        "| Dataset | Budget (KiB) | Reference | Categories | Delta AUPRO | Category range | Delta image AUROC |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for row in macro:
        lines.append(
            f"| {row['dataset']} | {row['budget_bytes'] / 1024:g} | {row['reference']} | {row['category_count']} | "
            f"{fmt(row['aupro_0_3_delta'])} | {fmt(row['aupro_0_3_min_category_delta'])} to {fmt(row['aupro_0_3_max_category_delta'])} | {fmt(row['image_auroc_delta'])} |"
        )
    lines += [
        "",
        "## Exploratory alpha sensitivity",
        "",
        "Counterfactual choices minimize the same normal losses on the saved PCA grid.",
        "This post-hoc sensitivity does not replace the preset selector or choose a final test-best alpha.",
        "",
        "| Dataset | Budget (KiB) | Alpha | Macro AUPRO |",
        "|---|---:|---:|---:|",
    ]
    for row in alpha_macro:
        score = "undefined" if row["aupro_0_3"] is None else f"{row['aupro_0_3']:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['budget_bytes'] / 1024:g} | {row['reconstruction_weight']:g} | {score} |"
        )
    lines += [
        "",
        "## Coverage and interpretation",
        "",
        f"Runs: {len(inputs)}; categories: {len({r['category'] for r in rows})}; algorithm seeds: {sorted({r['seed'] for r in rows})}.",
        f"Feasible candidates evaluated: {len(candidates)}; pool-saturated candidates: {sum(r['pool_saturated'] for r in candidates)}.",
        "Seeds repeat one cached image split; pixels and seeds are not independent category replications.",
        "Oracle gaps are metric-specific post-hoc diagnostics within the original selector grid/geometry.",
        "Baseline helper dimensions are excluded. Test oracles never change persisted normal-only selections.",
        "See paired category/seed tables and oracle_gaps.csv before interpreting macro differences.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return macro
