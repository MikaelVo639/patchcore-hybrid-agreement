"""Two-phase experiment runner: persist normal-only choices before opening test."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .cache import FeatureCache, check_compatible
from .io import environment, peak_rss_bytes, sha256, write_csv, write_json
from .maps import resize_map, smooth_map
from .metrics import evaluate
from .model import BudgetMemory
from .selection import select_models


def predict_cache(model, cache, sigma=0):
    scores, maps, latencies = [], [], []
    # One warm-up image, excluded from per-image feature-to-score latency.
    model.score(cache.features[0].reshape(-1, model.input_dimension))
    for features in cache.features:
        started = perf_counter()
        patch_scores = model.score(features.reshape(-1, model.input_dimension))
        score = float(patch_scores.max())
        anomaly_map = smooth_map(
            resize_map(patch_scores.reshape(features.shape[:2]), cache.masks.shape[1:]), sigma
        )
        latencies.append(perf_counter() - started)
        scores.append(score)
        maps.append(anomaly_map)
    return (
        np.array(scores),
        np.stack(maps),
        {
            "feature_latency_median_ms": float(np.median(latencies) * 1000),
            "feature_latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
            "latency_scope": "batch-1 cached features through transform/search/map resize/smoothing; no encoder or disk I/O",
            "warmup_images": 1,
            "timed_images": len(latencies),
        },
    )


def run_experiment(
    build_path, calibration_path, test_path, output, config, evaluate_all_candidates=False
):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new path to preserve prior runs")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "status.json", {"status": "running", "started_utc": datetime.now(UTC).isoformat()}
    )
    try:
        config.validate()
        write_json(output / "config.json", config.to_dict())
        write_json(output / "environment.json", environment())
        build = FeatureCache.load(build_path, "build")
        calibration = FeatureCache.load(calibration_path, "calibration")
        check_compatible(build, calibration)
        provenance = {
            "config_sha256_before_test": sha256(output / "config.json"),
            "build_sha256": sha256(build_path),
            "calibration_sha256": sha256(calibration_path),
            "metadata": build.metadata,
        }
        selected = select_models(build.rows(), calibration.rows(), config)
        for candidate in selected.candidates:
            if candidate["feasible"]:
                candidate["model_sha256"] = hashlib.sha256(
                    selected.candidate_models[candidate["candidate_key"]].to_bytes()
                ).hexdigest()
        write_csv(output / "candidates.csv", selected.candidates)
        provenance["candidates_sha256_before_test"] = sha256(output / "candidates.csv")
        write_json(output / "selection_timing.json", selected.timing)
        for row in selected.selections:
            if row["status"] == "selected":
                path = output / "models" / f"{row['key']}.npz"
                selected.models[row["key"]].save(path, row["budget_bytes"])
                row["model_sha256"] = sha256(path)
        # This file is written before the test file is even opened.
        write_json(output / "selected_configs.json", selected.selections)
        candidate_manifest = []
        if evaluate_all_candidates:
            for row in selected.candidates:
                if row["feasible"]:
                    path = output / "candidate_models" / f"{row['candidate_key']}.npz"
                    selected.candidate_models[row["candidate_key"]].save(path, row["budget_bytes"])
                    candidate_manifest.append({**row, "model_sha256": sha256(path)})
            write_json(output / "candidate_manifest.json", candidate_manifest)
            provenance["candidate_manifest_sha256_before_test"] = sha256(
                output / "candidate_manifest.json"
            )
        provenance["selection_sha256_before_test"] = sha256(output / "selected_configs.json")
        provenance["test_opened_after_selection"] = True
        write_json(output / "provenance.json", provenance)
        test = FeatureCache.load(test_path, "test")
        check_compatible(build, calibration, test)
        provenance["test_sha256"] = sha256(test_path)
        write_json(output / "provenance.json", provenance)
        rows = []
        for selection in selected.selections:
            if selection["status"] != "selected":
                continue
            key = selection["key"]
            # Score the reloaded deployed artifact, rather than a richer in-memory model.
            model = BudgetMemory.load(output / "models" / f"{key}.npz")
            scores, maps, latency = predict_cache(model, test, config.map_smoothing_sigma)
            row = {
                "dataset": test.metadata.get("dataset", "unknown"),
                "category": test.metadata.get("category", "unknown"),
                "synthetic": bool(test.metadata.get("synthetic", False)),
                "benchmark_eligible_weights": test.metadata.get("benchmark_eligible_weights"),
                **selection,
                **evaluate(test.labels, scores, test.masks, maps),
                **latency,
            }
            rows.append(row)
            (output / "predictions").mkdir(exist_ok=True)
            np.savez(
                output / "predictions" / f"{key}.npz",
                image_ids=test.image_ids,
                image_scores=scores,
                maps=maps,
                labels=test.labels,
                masks=test.masks,
            )
        write_csv(output / "metrics.csv", rows)
        write_json(output / "metrics.json", rows)
        if evaluate_all_candidates:
            # Test labels enter only here, after every choice/artifact is persisted.
            # Helper dimensions for fixed/variance baselines remain tagged and are
            # excluded from the joint selector's oracle by the analysis tool.
            evaluated = {r["candidate_key"]: r for r in rows}
            candidate_metrics = []
            for candidate in candidate_manifest:
                key = candidate["candidate_key"]
                if key in evaluated:
                    metrics = {
                        k: evaluated[key][k]
                        for k in ("image_auroc", "pixel_auroc", "aupro_0_3", "aupro_connectivity")
                    }
                else:
                    model = BudgetMemory.load(output / "candidate_models" / f"{key}.npz")
                    scores, maps, _ = predict_cache(model, test, config.map_smoothing_sigma)
                    metrics = evaluate(test.labels, scores, test.masks, maps)
                candidate_metrics.append(
                    {
                        "dataset": test.metadata["dataset"],
                        "category": test.metadata["category"],
                        **candidate,
                        **metrics,
                        "analysis_scope": "post-hoc only; never used for selection",
                    }
                )
            write_json(output / "candidate_metrics.json", candidate_metrics)
            write_csv(output / "candidate_metrics.csv", candidate_metrics)
        write_json(
            output / "status.json",
            {
                "status": "complete",
                "evaluated_models": len(rows),
                "evaluated_candidates": len(candidate_manifest),
                "infeasible_selections": sum(
                    r["status"] == "infeasible" for r in selected.selections
                ),
                "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
                "memory_scope": "Lifetime process peak, not isolated per method; GPU memory not measured",
                "completed_utc": datetime.now(UTC).isoformat(),
            },
        )
        return rows
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise


def reevaluate_predictions(path):
    with np.load(path, allow_pickle=False) as predictions:
        return evaluate(
            predictions["labels"],
            predictions["image_scores"],
            predictions["masks"],
            predictions["maps"],
        )


def load_metrics(directory):
    return json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
