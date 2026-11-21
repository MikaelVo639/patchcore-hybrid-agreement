"""Post-hoc oracle analysis over every feasible candidate. Never feeds back into selection.

The oracle re-creates all candidates from the run's own config and normal caches, proves
that the re-created selections reproduce the recorded model hashes, and only then opens
the test cache. Everything it writes goes under ``<run>/oracle/``; the selection files are
hashed before and after to show they were not modified.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .audit import read_candidates
from .cache import FeatureCache, check_compatible
from .io import sha256, write_csv, write_json
from .metrics import evaluate
from .runner import predict_cache
from .selection import ExperimentConfig, criterion_for, eligible, select_models

METRICS = ("image_auroc", "pixel_auroc", "aupro_0_3")


def _better(value, best):
    return value is not None and (best is None or value > best)


def _best(rows, metric):
    best = None
    for row in rows:
        if _better(row.get(metric), None if best is None else best[metric]):
            best = row
    return best


def oracle_run(run_dir, build_path, calibration_path, test_path):
    run_dir = Path(run_dir)
    output = run_dir / "oracle"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Oracle output already exists; keep it or move it before rerunning")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Oracle analysis requires a completed run (selection and test done)")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    selection_file = run_dir / "selected_configs.json"
    selection_hash = sha256(selection_file)
    if selection_hash != provenance["selection_sha256_before_test"]:
        raise ValueError("selected_configs.json changed after selection; oracle refused")
    for filename, field in (
        ("config.json", "config_sha256_before_test"),
        ("candidates.csv", "candidates_sha256_before_test"),
    ):
        if field not in provenance:
            raise ValueError(
                "Run lacks pre-test config/candidate provenance; use its original code or rerun into a new directory"
            )
        if sha256(run_dir / filename) != provenance[field]:
            raise ValueError(f"{filename} changed after selection; oracle refused")
    recorded_candidates = {
        r["candidate_key"]: r for r in read_candidates(run_dir / "candidates.csv") if r["feasible"]
    }
    for path, name in ((build_path, "build"), (calibration_path, "calibration")):
        if sha256(path) != provenance[f"{name}_sha256"]:
            raise ValueError(f"{name} cache differs from the one used by this run")

    config = ExperimentConfig.from_dict(
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    )
    recorded = json.loads(selection_file.read_text(encoding="utf-8"))
    build = FeatureCache.load(build_path, "build")
    calibration = FeatureCache.load(calibration_path, "calibration")
    check_compatible(build, calibration)
    # Normal-only re-creation; the test cache has not been opened yet.
    result = select_models(build.rows(), calibration.rows(), config)
    if result.candidate_models.keys() != recorded_candidates.keys():
        raise RuntimeError("Candidate grid did not reproduce; oracle refused")
    for key, model in result.candidate_models.items():
        if hashlib.sha256(model.to_bytes()).hexdigest() != recorded_candidates[key].get(
            "model_sha256"
        ):
            raise RuntimeError(
                f"Candidate {key} did not reproduce the pre-test model; oracle refused"
            )
    recreated = {r["key"]: r for r in result.selections}
    for row in recorded:
        if row["status"] != "selected":
            continue
        again = recreated.get(row["key"])
        if again is None or again["status"] != "selected":
            raise RuntimeError(f"{row['key']} was not re-selected; oracle refused")
        digest = hashlib.sha256(result.models[row["key"]].to_bytes()).hexdigest()
        if digest != row.get("model_sha256"):
            raise RuntimeError(
                f"{row['key']} did not reproduce the recorded model; oracle refused "
                "(check code version, config and platform)"
            )

    if sha256(test_path) != provenance.get("test_sha256"):
        raise ValueError("test cache differs from the one used by this run")
    test = FeatureCache.load(test_path, "test")
    check_compatible(build, calibration, test)
    candidate_rows = []
    for row in result.candidates:
        if not row["feasible"]:
            continue
        model = result.candidate_models[row["candidate_key"]]
        scores, maps, _ = predict_cache(model, test, config.map_smoothing_sigma)
        candidate_rows.append(
            {
                "candidate_key": row["candidate_key"],
                "budget_bytes": row["budget_bytes"],
                "variant": row["variant"],
                "dimension": row["dimension"],
                "bank_count": row["bank_count"],
                "artifact_bytes": row["artifact_bytes"],
                "binding_limit": row["binding_limit"],
                "reconstruction": row["reconstruction"],
                "coverage": row["coverage"],
                "joint": row["joint"],
                **evaluate(test.labels, scores, test.masks, maps),
            }
        )
    by_key = {r["candidate_key"]: r for r in candidate_rows}

    # Consistency: the selected candidates must reproduce the run's own test metrics.
    run_metrics = {
        r["key"]: r for r in json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    }
    mismatches = []
    gap_rows = []
    variance_dim = result.candidates[0]["variance_dimension"] if result.candidates else None
    for selection in result.selections:
        if selection["status"] != "selected":
            continue
        chosen = by_key[selection["candidate_key"]]
        reference = run_metrics.get(selection["key"], {})
        for metric in METRICS:
            if reference.get(metric) != chosen[metric]:
                mismatches.append(f"{selection['key']}.{metric}")
        budget_rows = [
            by_key[r["candidate_key"]]
            for r in result.candidates
            if r["budget_bytes"] == selection["budget_bytes"] and r["feasible"]
        ]
        method_keys = {
            r["candidate_key"]
            for r in eligible(
                selection["method"],
                [r for r in result.candidates if r["budget_bytes"] == selection["budget_bytes"]],
                config,
                variance_dim,
            )
        }
        method_rows = [r for r in budget_rows if r["candidate_key"] in method_keys]
        gap = {
            "key": selection["key"],
            "method": selection["method"],
            "budget_bytes": selection["budget_bytes"],
            "criterion": criterion_for(selection["method"]),
            "selected_candidate": selection["candidate_key"],
            "eligible_candidates": len(method_rows),
        }
        for metric in METRICS:
            selected_value = chosen[metric]
            best_method = _best(method_rows, metric)
            best_any = _best(budget_rows, metric)
            gap[f"selected_{metric}"] = selected_value
            gap[f"oracle_{metric}"] = best_method and best_method[metric]
            gap[f"oracle_candidate_{metric}"] = best_method and best_method["candidate_key"]
            gap[f"gap_{metric}"] = (
                None
                if selected_value is None or best_method is None
                else best_method[metric] - selected_value
            )
            gap[f"best_any_{metric}"] = best_any and best_any[metric]
            gap[f"best_any_candidate_{metric}"] = best_any and best_any["candidate_key"]
        gap_rows.append(gap)
    if mismatches:
        raise RuntimeError(f"Oracle metrics disagree with the run's metrics: {mismatches[:5]}")

    write_csv(output / "candidate_metrics.csv", candidate_rows)
    write_csv(output / "oracle_gap.csv", gap_rows)
    if sha256(selection_file) != selection_hash:
        raise RuntimeError("selected_configs.json changed during oracle analysis")
    write_json(
        output / "oracle.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "evaluated_candidates": len(candidate_rows),
            "selection_sha256_unchanged": selection_hash,
            "selected_models_reproduced": True,
            "selected_metrics_reproduced": True,
            "gap_definition": (
                "oracle_<metric> is the best test value among candidates that the method was "
                "allowed to choose at that budget; gap = oracle - selected (>= 0). best_any_* "
                "uses every feasible candidate at that budget."
            ),
            "use": (
                "Post-hoc analysis only. Do not change the selector, alpha, D candidates, budgets "
                "or caps from these numbers without labeling the change exploratory."
            ),
        },
    )
    return candidate_rows, gap_rows
