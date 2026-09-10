"""Command line interface.

Four pipeline stages, ``generate``, ``bench``, ``score`` and ``report``, plus
``run`` which chains the first three and ``prefetch`` which populates the cache
before any of them. Then ``info`` to see what the current environment can do,
``verify-testset`` to check test-set provenance against ALMA's own files, and
``suite-directions`` and ``run-dir`` so a batch script can resolve what it
needs without parsing configs itself, ``calibration`` to build the pruning and
repair data, and ``overlap`` to compare subnetworks with each other.

Configuration comes from YAML files. ``--set`` applies dotted-path overrides on
top, which change the run identity as they should: an override is a different
experiment, not a variant of the same one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from mnlp_eval import __version__
from mnlp_eval.config import (
    BenchSpec,
    ConfigError,
    MetricsSpec,
    ModelSpec,
    RunConfig,
    SuiteSpec,
    load_yaml_config,
)
from mnlp_eval.languages import ALMA_DIRECTIONS
from mnlp_eval.runspec import RunPaths, discover_runs

DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_METRICS_CONFIG = Path("configs/metrics/default.yaml")


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------
# Configuration assembly
# --------------------------------------------------------------------------


def _apply_overrides(
    payload: dict[str, Any], overrides: list[str], *, allowed_roots: set[str] | None = None
) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides, parsing values as YAML scalars.

    ``allowed_roots`` guards the first path segment. Without it, an override
    the command does not consume was written into an unread key and dropped in
    silence, so ``--set decode.num_beams=1`` (missing the ``suite.`` prefix)
    left a beam-5 run wearing a legitimate-looking run id.
    """
    for override in overrides:
        if "=" not in override:
            msg = f"malformed override {override!r}; expected the form key.path=value"
            raise ConfigError(msg)
        path, _, raw = override.partition("=")
        keys = [part for part in path.strip().split(".") if part]
        if not keys:
            msg = f"malformed override {override!r}; the key path is empty"
            raise ConfigError(msg)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            msg = f"cannot parse override value {raw!r}: {exc}"
            raise ConfigError(msg) from exc
        if allowed_roots is not None and keys[0] not in allowed_roots:
            expected = ", ".join(sorted(allowed_roots))
            msg = (
                f"override {override!r} targets {keys[0]!r}, which this command does not "
                f"read. Valid roots here are: {expected}"
            )
            raise ConfigError(msg)
        cursor: dict[str, Any] = payload
        for key in keys[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            cursor = existing
        cursor[keys[-1]] = value
    return payload


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    model_payload = load_yaml_config(args.model)
    suite_payload = load_yaml_config(args.suite)
    combined = {"model": model_payload, "suite": suite_payload}
    roots = {"model", "suite"}
    if getattr(args, "bench", "__absent__") != "__absent__" or hasattr(args, "no_bench"):
        # `run` and `bench` also accept bench overrides in the same --set list.
        roots.add("bench")
    _apply_overrides(combined, list(args.set or []), allowed_roots=roots)

    if getattr(args, "limit", None) is not None:
        combined["suite"].setdefault("data", {})["limit"] = args.limit
    if getattr(args, "directions", None):
        combined["suite"].setdefault("data", {})["directions"] = [
            item.strip() for item in args.directions.split(",") if item.strip()
        ]

    return RunConfig(
        model=ModelSpec.from_dict(combined["model"]),
        suite=SuiteSpec.from_dict(combined["suite"]),
        output_root=Path(args.runs_root),
    )


def _resolve_run_paths(args: argparse.Namespace) -> list[RunPaths]:
    if getattr(args, "run", None):
        paths = RunPaths(Path(args.run))
        if not paths.manifest.is_file():
            msg = f"no manifest at {paths.manifest}"
            raise FileNotFoundError(msg)
        return [paths]
    return list(discover_runs(Path(args.runs_root)))


def _build_bench_spec(args: argparse.Namespace, config: RunConfig | None = None) -> BenchSpec:
    """Build a bench spec from its config file plus ``--set bench.*`` overrides."""
    payload = load_yaml_config(args.bench) if getattr(args, "bench", None) else {}
    combined: dict[str, Any] = {"bench": payload}
    _apply_overrides(
        combined,
        list(getattr(args, "set", None) or []),
        allowed_roots={"bench", "model", "suite"},
    )
    payload = combined.get("bench") or {}
    if getattr(args, "bench_direction", None):
        payload["direction"] = args.bench_direction
    requested = "direction" in payload
    spec = BenchSpec.from_dict(payload)
    if config is None or spec.direction in config.suite.data.directions:
        return spec
    if requested:
        msg = (
            f"bench direction {spec.direction!r} is not in suite "
            f"{config.suite.name!r}, which covers {', '.join(config.suite.data.directions)}"
        )
        raise ConfigError(msg)
    # Only the default was out of range, so pick a direction the suite covers
    # rather than failing on a value nobody asked for.
    fallback = config.suite.data.directions[0]
    print(
        f"bench: suite {config.suite.name!r} does not cover the default direction "
        f"{spec.direction}, benchmarking {fallback} instead",
        file=sys.stderr,
    )
    return BenchSpec.from_dict({**payload, "direction": fallback})


def _load_metrics(path: str | None, overrides: list[str] | None = None) -> MetricsSpec:
    """Load a metrics config, applying ``--set`` overrides on top.

    Overrides are useful for fitting a large metric model onto a small GPU
    without editing a committed config, for example
    ``--set comet_batch_size=8``.
    """
    if path is not None:
        payload = load_yaml_config(path)
    elif DEFAULT_METRICS_CONFIG.is_file():
        payload = load_yaml_config(DEFAULT_METRICS_CONFIG)
    else:
        payload = {}
    if overrides:
        payload = _apply_overrides(payload, list(overrides))
    return MetricsSpec.from_dict(payload)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def command_info(args: argparse.Namespace) -> int:
    from mnlp_eval.data import available_datasets
    from mnlp_eval.metrics.base import group_availability
    from mnlp_eval.models import available_loaders
    from mnlp_eval.prompts import available_prompts

    print(f"mnlp-eval {__version__}")
    print(f"python     {sys.version.split()[0]} at {sys.executable}")
    print()
    print(f"loaders    {', '.join(available_loaders())}")
    print(f"prompts    {', '.join(available_prompts())}")
    print(f"datasets   {', '.join(available_datasets())}")
    print()
    print("metric groups in this environment:")
    for group, (served, reason) in group_availability().items():
        status = "available" if served else "unavailable"
        print(f"  {group:9} {status}")
        if reason:
            print(f"            {reason}")
    print()
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(
                f"device     {props.name}, "
                f"{props.total_memory / 1024**3:.1f} GiB, torch {torch.__version__}"
            )
        else:
            print(f"device     cpu only, torch {torch.__version__}")
    except ImportError:
        print("device     torch not installed in this environment")

    runs = list(discover_runs(Path(args.runs_root)))
    print(f"runs       {len(runs)} under {args.runs_root}")
    for paths in runs:
        manifest = paths.read_manifest()
        del manifest
        done = ", ".join(
            stage for stage in ("generate", "bench", "score") if paths.stage_completed(stage)
        )
        pending = sorted(set(paths.suite_directions()) - paths.generated_directions())
        note = done or "no completed stages"
        if pending:
            note += f"; awaiting {', '.join(pending)}"
        print(f"           {paths.root.name}  [{note}]")
    return 0


def command_prefetch(args: argparse.Namespace) -> int:
    from mnlp_eval.prefetch import prefetch

    models = [ModelSpec.from_dict(load_yaml_config(path)) for path in args.models or []]
    suites = [SuiteSpec.from_dict(load_yaml_config(path)) for path in args.suites or []]
    metrics = None if args.no_metrics else _load_metrics(args.metrics)
    if not models and not suites and metrics is None:
        return _fail("nothing to prefetch; pass --models, --suites or a metrics config")
    report = prefetch(models, suites, metrics)
    if args.json:
        print(json.dumps(report, indent=2))
    failed = any(not value.get("ok") for section in report.values() for value in section.values())
    return 1 if failed and args.strict else 0


def command_generate(args: argparse.Namespace) -> int:
    from mnlp_eval.generate import generate

    config = _build_run_config(args)
    generate(
        config,
        overwrite=args.overwrite,
        deterministic_check=args.deterministic_check,
        only_directions=args.only_direction,
    )
    print(config.directory)
    return 0


def command_bench(args: argparse.Namespace) -> int:
    from mnlp_eval.bench import bench_run

    if args.run:
        # The run's own manifest is authoritative here: benchmarking a model
        # under different settings than it was generated with would produce
        # efficiency numbers that do not describe the run they sit next to.
        paths = RunPaths(Path(args.run))
        config = RunConfig.from_manifest(paths.read_manifest(), Path(args.run).parent)
        # Act on the directory given, not on one recomputed from the slug,
        # so that a renamed or archived run is benchmarked in place.
        bench_run(config, _build_bench_spec(args, config), paths=paths)
        print(paths.bench)
        return 0

    config = _build_run_config(args)
    bench_run(config, _build_bench_spec(args, config))
    print(config.directory / "bench.json")
    return 0


def command_score(args: argparse.Namespace) -> int:
    from mnlp_eval.score import score_run

    metrics = _load_metrics(args.metrics, getattr(args, "set", None))
    groups = (
        [item.strip() for item in args.groups.split(",") if item.strip()] if args.groups else None
    )
    targets = _resolve_run_paths(args)
    if not targets:
        return _fail(f"no runs found under {args.runs_root}")
    failures = 0
    for paths in targets:
        print(f"scoring {paths.root.name}", file=sys.stderr)
        try:
            score_run(
                paths,
                metrics,
                groups=groups,
                overwrite=args.overwrite,
                allow_partial=args.allow_partial,
            )
        except RuntimeError as exc:
            # One unfinished run must not abort scoring for the others, which
            # is the normal case while a sweep is still generating.
            print(f"  skipped: {exc}", file=sys.stderr)
            failures += 1
    if failures and failures == len(targets):
        return _fail(f"every run under {args.runs_root} was skipped")
    return 0


def command_report(args: argparse.Namespace) -> int:
    from mnlp_eval.report import build_report

    metrics = _load_metrics(args.metrics, getattr(args, "set", None))
    result = build_report(
        Path(args.runs_root),
        Path(args.out),
        baseline=args.baseline,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        bootstrap_samples=metrics.bootstrap_samples,
        bootstrap_seed=metrics.bootstrap_seed,
        significance=not args.no_significance,
        plots=not args.no_plots,
    )
    if not result.files:
        return _fail("no report produced")
    print(Path(args.out) / "report.md")
    return 0


def command_run(args: argparse.Namespace) -> int:
    from mnlp_eval.bench import bench_run
    from mnlp_eval.generate import generate
    from mnlp_eval.score import score_run

    config = _build_run_config(args)
    # Resolve every configuration before generating. A bad metrics path or an
    # invalid bench spec used to surface after generation finished, which for
    # the six-direction suite is three to six A100-hours before a
    # missing-file error.
    bench_spec = None if args.no_bench else _build_bench_spec(args, config)
    metrics = _load_metrics(args.metrics)
    groups = (
        [item.strip() for item in args.groups.split(",") if item.strip()] if args.groups else None
    )

    generate(
        config,
        overwrite=args.overwrite,
        deterministic_check=args.deterministic_check,
        only_directions=args.only_direction,
    )
    paths = RunPaths.for_config(config)
    if bench_spec is not None:
        bench_run(config, bench_spec)
    score_run(
        paths,
        metrics,
        groups=groups,
        overwrite=args.overwrite,
        allow_partial=args.only_direction is not None,
    )
    print(config.directory)
    return 0


def command_suite_directions(args: argparse.Namespace) -> int:
    """Print a suite's directions, one per line.

    Exists so a batch script can index into them by array task id without
    passing a comma-separated list through ``sbatch --export``, where commas
    are the delimiter between assignments and silently truncate the value.
    """
    suite = SuiteSpec.from_dict(load_yaml_config(args.suite))
    for direction in suite.data.directions:
        print(direction)
    return 0


def command_run_dir(args: argparse.Namespace) -> int:
    """Print the run directory a model and suite resolve to.

    Lets a batch script name one run exactly, rather than a scoring job acting
    on every directory it can find, including other models' runs that are
    still generating.
    """
    print(_build_run_config(args).directory)
    return 0


def command_calibration(args: argparse.Namespace) -> int:
    """Build the calibration and repair data the pruning runs share."""
    from mnlp_eval.data.calibration import CalibrationSpec, build_calibration_set

    payload = load_yaml_config(args.spec)
    _apply_overrides(payload, list(getattr(args, "set", None) or []))
    spec = CalibrationSpec.from_dict(payload)
    manifest = build_calibration_set(
        spec,
        Path(args.out),
        contamination_check=None if args.no_contamination_check else args.contamination_check,
    )
    contamination = manifest.get("contamination") or {}
    collisions = contamination.get("total_collisions")
    if collisions:
        # Not a warning. A calibration set that overlaps the test set makes
        # every quality number downstream of it indefensible.
        return _fail(
            f"{collisions} calibration segment(s) also appear in "
            f"{contamination.get('dataset')}. This set must not be used."
        )
    unchecked = sorted(
        direction
        for direction, entry in (contamination.get("directions") or {}).items()
        if not entry.get("checked")
    )
    if unchecked:
        print(
            f"warning: contamination unchecked for {', '.join(unchecked)}",
            file=sys.stderr,
        )
    print(
        f"{spec.total_segments} segments across {len(spec.directions)} direction(s), "
        f"fingerprint {manifest['fingerprint']}",
        file=sys.stderr,
    )
    print(Path(args.out) / spec.name)
    return 0


def command_overlap(args: argparse.Namespace) -> int:
    """Compare subnetwork descriptors and write the overlap analysis."""
    from mnlp_eval.analysis import compare_subnetworks, load_subnetworks, overlap_report
    from mnlp_eval.artifacts import atomic_write_json, atomic_write_text

    candidates: list[str | Path] = list(args.subnetwork or [])
    for directory in args.subnetwork_dir or []:
        found = sorted(Path(directory).glob("*.json"))
        if not found:
            return _fail(f"no .json descriptors under {directory}")
        candidates.extend(found)
    if len(candidates) < 2:
        return _fail(
            "overlap needs at least two descriptors; pass --subnetwork twice or "
            "--subnetwork-dir once"
        )

    analysis = compare_subnetworks(load_subnetworks(candidates))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "overlap.md", overlap_report(analysis))
    atomic_write_json(out_dir / "overlap.json", analysis)
    for problem in analysis["problems"]:
        print(f"warning: {problem}", file=sys.stderr)
    print(out_dir / "overlap.md")
    return 0


def command_verify_testset(args: argparse.Namespace) -> int:
    from mnlp_eval.verify import verify_testset

    ok = verify_testset(
        dataset=args.dataset,
        directions=[item.strip() for item in args.directions.split(",") if item.strip()],
        reference_root=args.reference_root,
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_model_suite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="path to a model YAML config")
    parser.add_argument("--suite", required=True, help="path to a suite YAML config")
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="dotted-path override, for example --set suite.decode.batch_size=2",
    )
    parser.add_argument("--limit", type=int, help="cap segments per direction")
    parser.add_argument("--directions", help="comma-separated directions, overriding the suite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnlp-eval",
        description="Quality and efficiency evaluation for compressed translation models.",
    )
    parser.add_argument("--version", action="version", version=f"mnlp-eval {__version__}")
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help=f"directory holding run directories (default: {DEFAULT_RUNS_ROOT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show what this environment can do")
    info.set_defaults(handler=command_info)

    fetch = subparsers.add_parser(
        "prefetch", help="download models, datasets and metric checkpoints into the local cache"
    )
    fetch.add_argument("--models", nargs="*", help="model config paths")
    fetch.add_argument("--suites", nargs="*", help="suite config paths")
    fetch.add_argument("--metrics", help="metrics config path")
    fetch.add_argument("--no-metrics", action="store_true", help="skip metric checkpoints")
    fetch.add_argument("--json", action="store_true", help="print the report as JSON")
    fetch.add_argument(
        "--strict", action="store_true", help="exit non-zero if anything could not be fetched"
    )
    fetch.set_defaults(handler=command_prefetch)

    gen = subparsers.add_parser("generate", help="translate a suite and write a run directory")
    _add_model_suite_arguments(gen)
    gen.add_argument("--overwrite", action="store_true", help="regenerate existing directions")
    gen.add_argument(
        "--deterministic-check",
        type=int,
        default=0,
        metavar="N",
        help="re-translate N segments at batch size 1 and report the disagreement rate",
    )
    gen.add_argument(
        "--only-direction",
        action="append",
        metavar="PAIR",
        help=(
            "generate only this direction of the suite, leaving the rest to another "
            "process. Repeatable. Used to shard one run across a Slurm array: unlike "
            "--directions it does not change the suite, so every task contributes to "
            "the same run instead of creating a one-direction run of its own"
        ),
    )
    gen.set_defaults(handler=command_generate)

    bench = subparsers.add_parser("bench", help="measure efficiency")
    bench.add_argument("--run", help="an existing run directory")
    bench.add_argument("--model", help="model config, if --run is not given")
    bench.add_argument("--suite", help="suite config, if --run is not given")
    bench.add_argument("--bench", help="bench config path")
    bench.add_argument("--bench-direction", help="direction to benchmark")
    bench.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="dotted-path override, for example --set bench.subset_size=32",
    )
    bench.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    bench.add_argument("--directions", help=argparse.SUPPRESS)
    bench.set_defaults(handler=command_bench)

    score = subparsers.add_parser("score", help="score one or every run")
    score.add_argument("--run", help="a single run directory; default is every run")
    score.add_argument("--groups", help="comma-separated metric groups")
    score.add_argument("--metrics", help="metrics config path")
    score.add_argument("--overwrite", action="store_true", help="rescore existing groups")
    score.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "score a run whose generation is incomplete. Off by default: averaging "
            "over the directions that happen to be on disk produces a macro average "
            "covering fewer directions than the table claims"
        ),
    )
    score.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="metrics-config override, for example --set comet_batch_size=8",
    )
    score.set_defaults(handler=command_score)

    report = subparsers.add_parser("report", help="aggregate runs into tables and plots")
    report.add_argument("--out", default=str(DEFAULT_REPORT_DIR), help="output directory")
    report.add_argument("--baseline", help="model name to compare everything against")
    report.add_argument("--metrics", help="metrics config path, for bootstrap settings")
    report.add_argument("--formats", default="md,csv", help="comma-separated: md, csv, tex")
    report.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="metrics-config override, for example --set bootstrap_samples=200",
    )
    report.add_argument("--no-significance", action="store_true")
    report.add_argument("--no-plots", action="store_true")
    report.set_defaults(handler=command_report)

    chain = subparsers.add_parser("run", help="generate, then bench, then score")
    _add_model_suite_arguments(chain)
    chain.add_argument("--overwrite", action="store_true")
    chain.add_argument("--deterministic-check", type=int, default=0, metavar="N")
    chain.add_argument("--bench", help="bench config path")
    chain.add_argument("--bench-direction", help="direction to benchmark")
    chain.add_argument("--no-bench", action="store_true", help="skip the efficiency stage")
    chain.add_argument("--groups", help="comma-separated metric groups")
    chain.add_argument("--metrics", help="metrics config path")
    chain.add_argument("--only-direction", action="append", metavar="PAIR")
    chain.set_defaults(handler=command_run)

    directions = subparsers.add_parser(
        "suite-directions", help="print a suite's directions, one per line"
    )
    directions.add_argument("--suite", required=True, help="path to a suite YAML config")
    directions.set_defaults(handler=command_suite_directions)

    run_dir = subparsers.add_parser(
        "run-dir", help="print the run directory a model and suite resolve to"
    )
    _add_model_suite_arguments(run_dir)
    run_dir.set_defaults(handler=command_run_dir)

    calibration = subparsers.add_parser(
        "calibration", help="build calibration and repair data from ALMA's training set"
    )
    calibration.add_argument("--spec", required=True, help="path to a calibration YAML config")
    calibration.add_argument(
        "--out", default="data/calibration", help="directory to write the set into"
    )
    calibration.add_argument(
        "--contamination-check",
        default="haoranxu/WMT22-Test",
        help="test set to check the drawn segments against",
    )
    calibration.add_argument(
        "--no-contamination-check",
        action="store_true",
        help="skip the check, for a machine with no access to the test set",
    )
    calibration.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override, for example --set segments_per_direction=256",
    )
    calibration.set_defaults(handler=command_calibration)

    overlap = subparsers.add_parser(
        "overlap", help="compare subnetwork descriptors against each other and against chance"
    )
    overlap.add_argument(
        "--subnetwork",
        action="append",
        metavar="PATH",
        help="a subnetwork descriptor. Repeatable; see docs/subnetworks.md",
    )
    overlap.add_argument(
        "--subnetwork-dir",
        action="append",
        metavar="DIR",
        help="a directory of descriptors, all of whose .json files are compared",
    )
    overlap.add_argument("--out", default=str(DEFAULT_REPORT_DIR), help="output directory")
    overlap.set_defaults(handler=command_overlap)

    verify = subparsers.add_parser(
        "verify-testset", help="check a Hub test set against ALMA's own files"
    )
    verify.add_argument("--dataset", default="haoranxu/WMT22-Test")
    verify.add_argument(
        "--directions",
        default=",".join(ALMA_DIRECTIONS),
        help="comma-separated directions to check",
    )
    verify.add_argument(
        "--reference-root",
        default=None,
        help="local ALMA checkout; downloaded from GitHub when omitted",
    )
    verify.set_defaults(handler=command_verify_testset)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bench" and not args.run and not (args.model and args.suite):
        return _fail("bench needs either --run, or both --model and --suite")
    try:
        handler: Any = args.handler
        return int(handler(args))
    except (ConfigError, FileNotFoundError, KeyError, ValueError) as exc:
        return _fail(str(exc).strip("'\""))
    except ImportError as exc:
        # A missing optional dependency. The bitsandbytes configs in this
        # repository need the `quant` extra, which the documented install does
        # not include, and this surfaced as a raw traceback naming package
        # metadata rather than the extra.
        return _fail(
            f"{exc}. A dependency is missing: install the extra this model needs, for "
            'example uv pip install -e ".[gen,quant,surface]". See docs/environments.md.'
        )
    except OSError as exc:
        # Covers the offline case: transformers raises OSError when it cannot
        # reach the Hub, and the 25-frame traceback named neither the cause nor
        # the fix.
        hint = ""
        if "couldn't connect" in str(exc) or "offline" in str(exc).lower():
            hint = (
                " If HF_HUB_OFFLINE=1 is set, the asset is not in HF_HOME: run "
                "slurm/prefetch.sh from a login node. Otherwise check network access."
            )
        return _fail(f"{type(exc).__name__}: {exc}{hint}")
    except TypeError as exc:
        # A config value of the wrong shape, for example `model.kwargs: 7`.
        return _fail(f"{exc}. Check the types in your configuration files.")
    except RuntimeError as exc:
        return _fail(str(exc))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
