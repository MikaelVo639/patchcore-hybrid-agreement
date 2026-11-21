"""Command-line workflows, with explicit paths and no implicit experiment overwrites."""

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_run
from .datasets import prepare_mvtec
from .oracle import oracle_run
from .report import build_report
from .runner import reevaluate_predictions, run_experiment
from .selection import ExperimentConfig
from .synthetic import make_demo


def _config(path, seed=None):
    config = ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    if seed is not None:
        config.seed = seed
    return config.validate()


def _run(cache, output, config, evaluate_all_candidates=False):
    cache = Path(cache)
    return run_experiment(
        cache / "build.npz",
        cache / "calibration.npz",
        cache / "test.npz",
        output,
        config,
        evaluate_all_candidates=evaluate_all_candidates,
    )


def parser():
    root = argparse.ArgumentParser(
        prog="pcbudget", description="Normal-only PatchCore-style N-D allocation experiments"
    )
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run a small CPU-only synthetic example and build a report"
    )
    demo.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--config", type=Path)
    split = commands.add_parser("prepare-mvtec", help="Create a disjoint image-level manifest")
    split.add_argument("--root", type=Path, required=True)
    split.add_argument("--category", required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--calibration-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)
    extract = commands.add_parser(
        "extract", help="Extract fixed-grid Torchvision features (optional image dependencies)"
    )
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--root", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument(
        "--backbone", choices=["resnet18", "wide_resnet50_2"], default="wide_resnet50_2"
    )
    extract.add_argument("--weights", choices=["IMAGENET1K_V1", "none"], default="IMAGENET1K_V1")
    extract.add_argument("--image-size", type=int, default=224)
    extract.add_argument("--resize-size", type=int, default=256)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--device", default="cpu")
    run = commands.add_parser("run", help="Select on normal caches, then evaluate on test")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--seed", type=int)
    run.add_argument(
        "--evaluate-all-candidates",
        action="store_true",
        help="Post-hoc test evaluation of the persisted candidate grid",
    )
    sweep = commands.add_parser(
        "sweep", help="Run several algorithm seeds against one fixed image split"
    )
    sweep.add_argument("--cache", type=Path, required=True)
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sweep.add_argument("--evaluate-all-candidates", action="store_true")
    report = commands.add_parser(
        "report", help="Aggregate completed runs and generate Markdown/SVG"
    )
    report.add_argument("--runs", type=Path, nargs="+", required=True)
    report.add_argument("--out", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Recompute metrics from one saved predictions.npz"
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    audit = commands.add_parser(
        "audit", help="Reload selected models, check them against B and summarize limits"
    )
    audit.add_argument("--run", type=Path, required=True)
    oracle = commands.add_parser(
        "oracle",
        help="Post-hoc: evaluate every feasible candidate of a completed run on test",
    )
    oracle.add_argument("--run", type=Path, required=True)
    oracle.add_argument("--cache", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("Demo output must be empty; choose a new --out")
            config = (
                _config(args.config, args.seed)
                if args.config
                else ExperimentConfig(
                    dimensions=[4, 8, 12],
                    fixed_dimension=8,
                    budgets_bytes=[4096, 8192],
                    seed=args.seed,
                    max_fit_patches=1000,
                    max_bank_pool=512,
                )
            )
            cache = make_demo(args.out / "cache", args.seed)
            rows = _run(cache, args.out / "run", config)
            build_report([args.out / "run"], args.out / "report")
            print(
                f"Synthetic demo complete: {len(rows)} models. Report: {args.out / 'report' / 'report.md'}"
            )
        elif args.command == "prepare-mvtec":
            manifest = prepare_mvtec(
                args.root, args.category, args.out, args.calibration_fraction, args.seed
            )
            counts = {
                split: sum(r["split"] == split for r in manifest["records"])
                for split in ("build", "calibration", "test")
            }
            print(json.dumps(counts, indent=2))
        elif args.command == "extract":
            from .features import extract_mvtec

            extract_mvtec(
                args.manifest,
                args.root,
                args.out,
                args.backbone,
                args.weights,
                args.image_size,
                args.resize_size,
                args.batch_size,
                args.device,
            )
            print(f"Feature caches saved: {args.out}")
        elif args.command == "run":
            rows = _run(
                args.cache, args.out, _config(args.config, args.seed), args.evaluate_all_candidates
            )
            print(f"Evaluated {len(rows)} models. Results: {args.out / 'metrics.csv'}")
        elif args.command == "sweep":
            if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
                raise ValueError("Seeds must be unique nonnegative integers")
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("Sweep output must be empty; choose a new --out")
            runs = []
            for seed in args.seeds:
                run = args.out / f"seed_{seed}"
                _run(args.cache, run, _config(args.config, seed), args.evaluate_all_candidates)
                runs.append(run)
                print(f"Completed algorithm seed {seed}")
            build_report(runs, args.out / "report")
            print(f"Sweep report: {args.out / 'report' / 'report.md'}")
        elif args.command == "report":
            build_report(args.runs, args.out)
            print(f"Report: {args.out / 'report.md'}")
        elif args.command == "evaluate":
            print(json.dumps(reevaluate_predictions(args.predictions), indent=2, allow_nan=False))
        elif args.command == "audit":
            result = audit_run(args.run)
            print(json.dumps(result["summary"], indent=2))
            print(f"Audit {result['status']}: {args.run / 'audit.md'}")
            for failure in result["failures"]:
                print(f"  {failure}", file=sys.stderr)
            if result["status"] != "pass":
                return 1
        elif args.command == "oracle":
            candidates, gaps = oracle_run(
                args.run,
                args.cache / "build.npz",
                args.cache / "calibration.npz",
                args.cache / "test.npz",
            )
            print(
                f"Oracle evaluated {len(candidates)} candidates, {len(gaps)} selections. "
                f"Post-hoc only: {args.run / 'oracle' / 'oracle_gap.csv'}"
            )
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
