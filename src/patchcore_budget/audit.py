"""Storage audit of a completed or partial run. Reads normal-side outputs only, never test."""

import csv
import json
from pathlib import Path

from .io import sha256, write_json
from .model import BudgetMemory


def _number(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def _truth(value):
    return str(value).lower() == "true"


def read_candidates(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["feasible"] = _truth(row["feasible"])
        for name in ("budget_bytes", "dimension", "bank_count", "pool_patches", "artifact_bytes"):
            row[name] = _number(row.get(name))
        if row["feasible"]:
            if row.get("binding_limit") in (None, ""):
                # Runs written before binding_limit existed: a full pool is the best evidence.
                row["binding_limit"] = (
                    "pool (inferred)"
                    if row["bank_count"] == row["pool_patches"]
                    else "budget (inferred)"
                )
            if row.get("unused_bytes") in (None, ""):
                row["unused_bytes"] = row["budget_bytes"] - row["artifact_bytes"]
            row["unused_bytes"] = _number(row["unused_bytes"])
    return rows


def audit_run(run_dir):
    """Reload every selected model and check its file against B and the recorded hash."""
    run_dir = Path(run_dir)
    selections = json.loads((run_dir / "selected_configs.json").read_text(encoding="utf-8"))
    candidates = read_candidates(run_dir / "candidates.csv")
    checks, failures = [], []
    for row in selections:
        if row["status"] != "selected":
            continue
        path = run_dir / "models" / f"{row['key']}.npz"
        check = {"key": row["key"], "budget_bytes": row["budget_bytes"], "path": str(path)}
        problems = []
        if not path.is_file():
            problems.append("model file missing")
        else:
            size = path.stat().st_size
            check["file_bytes"] = size
            if size > row["budget_bytes"]:
                problems.append(f"file exceeds budget by {size - row['budget_bytes']} bytes")
            if size != row["artifact_bytes"]:
                problems.append("file size differs from recorded artifact_bytes")
            if row.get("model_sha256") and sha256(path) != row["model_sha256"]:
                problems.append("sha256 differs from selected_configs.json")
            try:
                model = BudgetMemory.load(path)
                if model.bank.shape != (row["bank_count"], row["dimension"]):
                    problems.append(
                        f"bank shape {model.bank.shape} differs from recorded "
                        f"({row['bank_count']}, {row['dimension']})"
                    )
            except (ValueError, OSError, KeyError) as exc:
                problems.append(f"model does not reload: {exc}")
        check["status"] = "fail" if problems else "pass"
        check["problems"] = problems
        checks.append(check)
        failures += [f"{row['key']}: {p}" for p in problems]

    feasible = [r for r in candidates if r["feasible"]]
    summary = {
        "candidates": len(candidates),
        "feasible": len(feasible),
        "infeasible": len(candidates) - len(feasible),
        "pool_bound": sum(r["binding_limit"].startswith("pool") for r in feasible),
        "budget_bound": sum(r["binding_limit"].startswith("budget") for r in feasible),
        "selected_models_checked": len(checks),
        "infeasible_selections": sum(r["status"] == "infeasible" for r in selections),
    }
    result = {
        "status": "fail" if failures else "pass",
        "summary": summary,
        "failures": failures,
        "model_checks": checks,
        "scope": "Selected model files and candidates.csv only; the test cache is not read.",
    }
    write_json(run_dir / "audit.json", result)
    (run_dir / "audit.md").write_text(_markdown(result, candidates), encoding="utf-8")
    return result


def _markdown(result, candidates):
    s = result["summary"]
    lines = [
        "# Storage audit",
        "",
        f"**Status: {result['status'].upper()}** — {s['selected_models_checked']} selected models "
        f"reloaded; {s['feasible']}/{s['candidates']} candidates feasible; "
        f"{s['pool_bound']} limited by the pool, {s['budget_bound']} by B.",
        "",
        "A pool-bound candidate leaves budget unused: its N is set by `max_bank_pool` or the "
        "available build patches, not by B, so it does not test the N–D trade-off at that B.",
        "",
    ]
    if result["failures"]:
        lines += ["## Failures", ""] + [f"- {f}" for f in result["failures"]] + [""]
    for budget in sorted({r["budget_bytes"] for r in candidates}):
        lines += [
            f"## B = {budget:,} bytes",
            "",
            "| Variant | D | N | Limited by | Used | Unused bytes | Note |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
        for r in (c for c in candidates if c["budget_bytes"] == budget):
            if r["feasible"]:
                used = f"{r['artifact_bytes'] / budget:.1%}"
                lines.append(
                    f"| {r['variant']} | {r['dimension']} | {r['bank_count']:,} | "
                    f"{r['binding_limit']} | {used} | {r['unused_bytes']:,} | |"
                )
            else:
                lines.append(
                    f"| {r['variant']} | {r['dimension']} | — | infeasible | — | — | "
                    f"{r.get('reason', '')} |"
                )
        lines.append("")
    return "\n".join(lines)
