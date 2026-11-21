"""Normal-only allocation. This module deliberately has no test-label input."""

import math
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .model import BudgetMemory, max_bank_count
from .numerics import bank_order, matrix, nearest, sample_rows
from .projection import PCAFit

METHODS = ("raw", "fixed", "variance", "joint", "reconstruction", "coverage", "whitened_joint")
LOSS_METHODS = ("joint", "reconstruction", "coverage", "whitened_joint")


@dataclass
class ExperimentConfig:
    dimensions: list[int] = field(default_factory=lambda: [8, 16, 32])
    budgets_bytes: list[int] = field(default_factory=lambda: [8192, 16384, 32768])
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    fixed_dimension: int = 16
    variance_threshold: float = 0.95
    reconstruction_weight: float = 0.5
    whitening_regularization: float = 0.01
    max_fit_patches: int = 20000
    max_bank_pool: int = 10000
    max_calibration_patches: int = 10000
    sampler: str = "coreset"
    seed: int = 0
    map_smoothing_sigma: float = 0.0
    # Off by default; enabling it is a protocol decision (see docs/method_spec.md).
    include_raw_in_joint: bool = False
    # Relative tolerance for treating loss values as tied (floating-point rounding only).
    tie_tolerance: float = 1e-6

    def validate(self):
        for name in ("dimensions", "budgets_bytes"):
            values = getattr(self, name)
            if not values or any(type(x) is not int or x < 1 for x in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} cannot contain duplicates")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be nonempty and unique")
        if set(self.methods) - set(METHODS):
            raise ValueError(f"Unknown methods; choose from {METHODS}")
        for name in (
            "fixed_dimension",
            "max_fit_patches",
            "max_bank_pool",
            "max_calibration_patches",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 < self.variance_threshold <= 1:
            raise ValueError("variance_threshold must be in (0, 1]")
        if not 0 <= self.reconstruction_weight <= 1:
            raise ValueError("reconstruction_weight must be in [0, 1]")
        if not np.isfinite(self.whitening_regularization) or self.whitening_regularization <= 0:
            raise ValueError("whitening_regularization must be finite and positive")
        if self.sampler not in ("coreset", "random"):
            raise ValueError("sampler must be coreset or random")
        if not np.isfinite(self.map_smoothing_sigma) or self.map_smoothing_sigma < 0:
            raise ValueError("map_smoothing_sigma must be finite and nonnegative")
        if type(self.include_raw_in_joint) is not bool:
            raise ValueError("include_raw_in_joint must be true or false")
        if (
            isinstance(self.tie_tolerance, bool)
            or not isinstance(self.tie_tolerance, int | float)
            or not np.isfinite(self.tie_tolerance)
            or not 0 <= self.tie_tolerance < 1e-2
        ):
            raise ValueError("tie_tolerance must be finite and in [0, 0.01)")
        return self

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        return cls(**value).validate()

    def to_dict(self):
        return asdict(self)


@dataclass
class SelectionResult:
    models: dict
    candidates: list[dict]
    selections: list[dict]
    timing: dict
    # Every feasible candidate model, keyed by candidate_key. Only post-hoc oracle
    # analysis evaluates the unselected ones; selection itself never sees test data.
    candidate_models: dict = field(default_factory=dict)


def candidate_key(variant, dimension, budget):
    return f"{variant}_d{dimension}__b{budget}"


def eligible(method, rows, config, variance_dimension):
    """Feasible candidate rows a method may choose from, at one budget."""
    rows = [r for r in rows if r["feasible"]]
    if method == "raw":
        return [r for r in rows if r["variant"] == "raw"]
    if method in ("fixed", "variance"):
        target = config.fixed_dimension if method == "fixed" else variance_dimension
        return [r for r in rows if r["variant"] == "pca" and r["dimension"] == target]
    variant = "whitened" if method == "whitened_joint" else "pca"
    chosen = [r for r in rows if r["variant"] == variant and r["dimension"] in config.dimensions]
    if config.include_raw_in_joint and method != "whitened_joint":
        chosen += [r for r in rows if r["variant"] == "raw"]
    return chosen


def criterion_for(method):
    if method == "raw":
        return "coverage"
    if method in ("reconstruction", "coverage"):
        return method
    return "joint"


def choose(rows, criterion, tolerance):
    """Minimum loss; near-equal losses (relative tolerance) use fewer bytes, then smaller D."""
    best = min(r[criterion] for r in rows)
    limit = best + tolerance * max(abs(best), 1e-12)
    tied = [r for r in rows if r[criterion] <= limit]
    chosen = min(tied, key=lambda r: (r["artifact_bytes"], r["dimension"], r[criterion]))
    return chosen, len(tied)


def _calibration_losses(model, projected_cal, cal, mean, inverse, normal_scale, weight):
    distances, neighbors = nearest(projected_cal, model.bank)
    if inverse is None:
        reconstruction = 0.0
        coverage = float(distances.mean() / normal_scale)
    else:
        residual = cal.astype(np.float64) - (projected_cal.astype(np.float64) @ inverse + mean)
        delta = (projected_cal - model.bank[neighbors]).astype(np.float64) @ inverse
        reconstruction = float(np.mean(np.sum(residual**2, axis=1)) / normal_scale)
        coverage = float(np.mean(np.sum(delta**2, axis=1)) / normal_scale)
    return {
        "reconstruction": reconstruction,
        "coverage": coverage,
        "joint": weight * reconstruction + (1 - weight) * coverage,
    }


def select_models(build, calibration, config):
    """Fit on build, choose on normal calibration. Feature rows have equal weight.

    The cache runner uses a fixed patch grid, so every image has equal total weight.
    If a calibration cap is active, it samples patch rows uniformly and records the cap.
    """
    config.validate()
    build, calibration = matrix(build, "build"), matrix(calibration, "calibration")
    if build.shape[1] != calibration.shape[1]:
        raise ValueError("Build and calibration feature dimensions differ")
    started = perf_counter()
    pca_started = perf_counter()
    pca = PCAFit.fit(build, config.max_fit_patches, config.seed)
    pca_seconds = perf_counter() - pca_started
    pool = sample_rows(build, config.max_bank_pool, config.seed + 1)
    cal = sample_rows(calibration, config.max_calibration_patches, config.seed + 2)
    scale_sample = sample_rows(build, config.max_fit_patches, config.seed).astype(np.float64)
    normal_scale = max(float(np.mean(np.sum((scale_sample - pca.mean) ** 2, axis=1))), 1e-12)
    variance_dim = pca.variance_dimension(config.variance_threshold)
    dims = set(config.dimensions)
    if "fixed" in config.methods:
        dims.add(config.fixed_dimension)
    if "variance" in config.methods:
        dims.add(variance_dim)
    variants = [("pca", d) for d in sorted(dims)]
    if "whitened_joint" in config.methods:
        variants += [("whitened", d) for d in sorted(set(config.dimensions))]
    if "raw" in config.methods or config.include_raw_in_joint:
        variants.insert(0, ("raw", build.shape[1]))

    rows_by_cell, models_by_cell = {}, {}
    ordering_total = 0.0
    for variant_index, (variant, dim) in enumerate(variants):
        base = {
            "selector_candidate": (
                (variant in ("pca", "whitened") and dim in config.dimensions)
                or (variant == "raw" and config.include_raw_in_joint)
            ),
            "variant": variant,
            "dimension": dim,
            "seed": config.seed,
            "sampler": config.sampler,
            "feasible": False,
            "pool_patches": len(pool),
            "calibration_patches": len(cal),
            "pca_fit_patches": pca.fitted_rows,
            "normal_scale": normal_scale,
            "variance_dimension": variance_dim,
        }
        transform_started = perf_counter()
        try:
            if variant == "raw":
                mean = np.empty(0, np.float32)
                forward = np.empty((0, 0), np.float32)
                projected_pool, projected_cal, inverse = pool, cal, None
            else:
                forward, inverse = pca.transform_for(
                    dim, variant == "whitened", config.whitening_regularization
                )
                mean = pca.mean
                projected_pool = (pool - mean) @ forward
                projected_cal = (cal - mean) @ forward
            transform_error = None
        except ValueError as exc:
            transform_error = str(exc)
        transform_seconds = perf_counter() - transform_started

        counts, budget_counts, errors = {}, {}, {}
        for budget in config.budgets_bytes:
            if transform_error:
                errors[budget] = transform_error
                continue
            try:
                budget_counts[budget] = max_bank_count(dim, mean, forward, budget)
                counts[budget] = min(budget_counts[budget], len(pool))
            except ValueError as exc:
                errors[budget] = str(exc)

        # One prefix-consistent order serves every budget of this variant.
        ordering_started = perf_counter()
        order = (
            bank_order(projected_pool, max(counts.values()), config.seed, config.sampler)
            if counts
            else None
        )
        ordering_seconds = perf_counter() - ordering_started
        ordering_total += ordering_seconds

        for budget in config.budgets_bytes:
            row = {
                "budget_bytes": budget,
                **base,
                "candidate_key": candidate_key(variant, dim, budget),
            }
            candidate_started = perf_counter()
            if budget in counts:
                model = BudgetMemory(projected_pool[order[: counts[budget]]], mean, forward)
                storage = model.storage()
                if storage["artifact_bytes"] > budget:
                    raise RuntimeError("Internal storage accounting mismatch")
                row.update(
                    feasible=True,
                    bank_count=len(model.bank),
                    budget_max_count=budget_counts[budget],
                    binding_limit="pool" if budget_counts[budget] > len(pool) else "budget",
                    pool_saturated=len(model.bank) == len(pool),
                    **storage,
                    unused_bytes=budget - storage["artifact_bytes"],
                    unused_budget_bytes=budget - storage["artifact_bytes"],
                    budget_utilization=storage["artifact_bytes"] / budget,
                    bank_fraction=storage["bank_bytes"] / budget,
                    transform_fraction=(storage["mean_bytes"] + storage["projection_bytes"])
                    / budget,
                    container_fraction=storage["container_bytes"] / budget,
                    **_calibration_losses(
                        model,
                        projected_cal,
                        cal,
                        mean,
                        inverse,
                        normal_scale,
                        config.reconstruction_weight,
                    ),
                )
                models_by_cell[(budget, variant_index)] = model
            else:
                row["reason"] = errors[budget]
            row["candidate_seconds"] = perf_counter() - candidate_started
            row["variant_transform_seconds"] = transform_seconds
            row["variant_ordering_seconds"] = ordering_seconds
            rows_by_cell[(budget, variant_index)] = row

    candidates, selections, models, candidate_models = [], [], {}, {}
    for budget in config.budgets_bytes:
        rows = [rows_by_cell[(budget, i)] for i in range(len(variants))]
        candidates += rows
        for i, row in enumerate(rows):
            if (budget, i) in models_by_cell:
                candidate_models[row["candidate_key"]] = models_by_cell[(budget, i)]
        for method in config.methods:
            key = f"{method}__b{budget}"
            pool_rows = eligible(method, rows, config, variance_dim)
            if not pool_rows:
                selections.append(
                    {
                        "key": key,
                        "method": method,
                        "budget_bytes": budget,
                        "status": "infeasible",
                        "reason": "No eligible candidate fits; no silent fallback",
                    }
                )
                continue
            criterion = criterion_for(method)
            chosen, tied = choose(pool_rows, criterion, config.tie_tolerance)
            models[key] = candidate_models[chosen["candidate_key"]]
            selections.append(
                {
                    **chosen,
                    "key": key,
                    "method": method,
                    "status": "selected",
                    "criterion": criterion,
                    "eligible_candidates": len(pool_rows),
                    "tied_candidates": tied,
                }
            )
    for row in candidates:
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"Nonfinite {name} for {row['candidate_key']}")
    return SelectionResult(
        models,
        candidates,
        selections,
        {
            "pca_seconds": pca_seconds,
            "ordering_seconds_total": ordering_total,
            "selection_total_seconds": perf_counter() - started,
            "note": (
                "Includes all candidates and calibration; excludes feature extraction and "
                "final artifact writes. Each variant computes one bank order for its largest "
                "feasible budget; variant_* columns repeat that shared cost on each budget row, "
                "so sum them once per variant, not per row."
            ),
        },
        candidate_models,
    )
