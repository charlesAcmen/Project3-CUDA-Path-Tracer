"""Reproducible schema-v2 benchmark matrix runner.

The runner never guesses which CSV is newest. The renderer prints an exact
``PROFILER_OUTPUT_DIR=...`` marker; that directory is schema-validated and
recorded in ``manifest.json`` before report generation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - Python < 3.11
    raise SystemExit("benchmark_runner.py requires Python 3.11+") from exc

import profiler_utils as pu


OUTPUT_MARKER = re.compile(r"^PROFILER_OUTPUT_DIR=(.+)$", re.MULTILINE)
VALID_ROLES = {"throughput", "detail", "counter"}
COMPACT_NAMES = {0: "off", 1: "global", 2: "thrust", 3: "shared"}
RNG_NAMES = {0: "lcg", 1: "halton"}


@dataclass(frozen=True)
class Experiment:
    sequence: int
    experiment_id: str
    label: str
    scene: Path
    compact: int
    sort: bool
    rng: int
    direct_lighting: bool
    rr_min_bounces: int | None
    roles: tuple[str, ...]
    postprocess: dict[str, Any]
    baseline: bool
    compare_to: str | None
    claim_class: str
    headline: bool


@dataclass(frozen=True)
class Job:
    experiment: Experiment
    role: str
    repetition: int


def _require(table: dict[str, Any], key: str, context: str) -> Any:
    if key not in table:
        raise ValueError(f"{context}: missing required key {key!r}")
    return table[key]


def _resolve(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_spec(path: str | Path) -> tuple[dict[str, Any], list[Experiment]]:
    spec_path = Path(path).resolve()
    data = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    runner = dict(data.get("runner", {}))
    root = spec_path.parent
    default_roles = tuple(runner.get("roles", ["throughput", "detail", "counter"]))
    if (not default_roles or not set(default_roles) <= VALID_ROLES
            or len(set(default_roles)) != len(default_roles)):
        raise ValueError(f"runner.roles must be chosen from {sorted(VALID_ROLES)}")

    defaults = dict(data.get("defaults", {}))
    default_post = {
        "bloom": {"enabled": False, "threshold": 1.0, "intensity": 0.5,
                  "radius": 10, "sigma": 5.0},
        "chromaticAberration": {"enabled": False, "intensity": 0.003},
        "vignette": {"enabled": False, "intensity": 0.5, "exponent": 2.0},
    }
    for section, values in defaults.get("postprocess", {}).items():
        if isinstance(values, dict) and isinstance(default_post.get(section), dict):
            default_post[section].update(values)
        else:
            default_post[section] = values
    default_rr = int(_require(defaults, "rr_min_bounces", "defaults"))

    experiments: list[Experiment] = []
    seen: set[str] = set()
    for index, raw in enumerate(data.get("experiment", [])):
        context = f"experiment[{index}]"
        experiment_id = str(_require(raw, "id", context))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise ValueError(f"{context}.id may contain only letters, digits, '_' and '-'")
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment id: {experiment_id}")
        seen.add(experiment_id)
        compact = int(raw.get("compact", defaults.get("compact", 3)))
        rng = int(raw.get("rng", defaults.get("rng", 0)))
        if compact not in COMPACT_NAMES:
            raise ValueError(f"{context}.compact must be 0, 1, 2, or 3")
        if rng not in RNG_NAMES:
            raise ValueError(f"{context}.rng must be 0 or 1")
        roles = tuple(raw.get("roles", default_roles))
        if (not roles or not set(roles) <= VALID_ROLES
                or len(set(roles)) != len(roles)):
            raise ValueError(f"{context}.roles must be chosen from {sorted(VALID_ROLES)}")
        postprocess = json.loads(json.dumps(default_post))
        for section, values in raw.get("postprocess", {}).items():
            if isinstance(values, dict) and isinstance(postprocess.get(section), dict):
                postprocess[section].update(values)
            else:
                postprocess[section] = values
        experiments.append(Experiment(
            sequence=index,
            experiment_id=experiment_id,
            label=str(raw.get("label", experiment_id)),
            scene=_resolve(root, str(_require(raw, "scene", context))),
            compact=compact,
            sort=bool(raw.get("sort", defaults.get("sort", False))),
            rng=rng,
            direct_lighting=bool(raw.get(
                "direct_lighting", defaults.get("direct_lighting", True))),
            rr_min_bounces=int(raw.get("rr_min_bounces", default_rr)),
            roles=roles,
            postprocess=postprocess,
            baseline=bool(raw.get("baseline", False)),
            compare_to=(str(raw["compare_to"]) if raw.get("compare_to") else None),
            claim_class=str(raw.get("claim_class", "implementation")),
            headline=bool(raw.get("headline", True)),
        ))
    if not experiments:
        raise ValueError("spec must contain at least one [[experiment]]")
    if sum(experiment.baseline for experiment in experiments) > 1:
        raise ValueError("at most one experiment may set baseline=true")
    ids = {experiment.experiment_id for experiment in experiments}
    for experiment in experiments:
        if experiment.compare_to is not None and experiment.compare_to not in ids:
            raise ValueError(
                f"{experiment.experiment_id}.compare_to references unknown experiment "
                f"{experiment.compare_to!r}")
        if experiment.compare_to == experiment.experiment_id:
            raise ValueError(f"{experiment.experiment_id}.compare_to cannot reference itself")
    return runner, experiments


def make_jobs(experiments: list[Experiment], repetitions: int,
              seed: int) -> list[Job]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    jobs: list[Job] = []
    # Randomize within each repeat so every configuration sees early/late
    # thermal states, while the repetition number remains auditable.
    for repetition in range(1, repetitions + 1):
        round_jobs = [Job(experiment, role, repetition)
                      for experiment in experiments for role in experiment.roles]
        rng.shuffle(round_jobs)
        jobs.extend(round_jobs)
    return jobs


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                text=True, check=True)
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _run_root(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    candidate = output_root / f"benchmark_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"benchmark_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _generated_config(job: Job, run_root: Path, warmup: int) -> dict[str, Any]:
    experiment = job.experiment
    config: dict[str, Any] = {
        "compactMethod": experiment.compact,
        "sortByMaterial": experiment.sort,
        "rngMode": experiment.rng,
        "directLighting": experiment.direct_lighting,
        "saveAt": [],
        **experiment.postprocess,
        "profiler": {
            "enabled": True,
            "warmup": warmup,
            "mode": "throughput" if job.role == "throughput" else "detail",
            "collectCounters": job.role == "counter",
            "outputDir": str((run_root / "runs").resolve()),
            "tag": f"{experiment.experiment_id}_{job.role}_r{job.repetition}",
        },
    }
    config["rrMinBounces"] = experiment.rr_min_bounces
    return config


def _run_job(exe: Path, job: Job, run_root: Path, warmup: int,
             timeout: int, repo_root: Path) -> dict[str, Any]:
    key = f"{job.experiment.experiment_id}__{job.role}__r{job.repetition}"
    config_path = run_root / "configs" / f"{key}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(_generated_config(job, run_root, warmup), indent=2) + "\n",
        encoding="utf-8")
    command = [str(exe), str(job.experiment.scene), f"--config={config_path}"]
    started = datetime.now(timezone.utc)
    try:
        result = subprocess.run(command, cwd=repo_root, capture_output=True,
                                text=True, timeout=timeout)
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        error = None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = None
        error = f"timeout after {timeout}s"
    except OSError as exc:
        stdout, stderr, returncode, error = "", "", None, str(exc)

    log_path = run_root / "logs" / f"{key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND: {subprocess.list2cmdline(command)}\n"
        f"RETURN_CODE: {returncode}\nERROR: {error or ''}\n\n"
        f"--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}", encoding="utf-8")

    record: dict[str, Any] = {
        "key": key, "experiment_id": job.experiment.experiment_id,
        "label": job.experiment.label, "role": job.role,
        "repetition": job.repetition, "baseline": job.experiment.baseline,
        "command": command, "config": str(config_path), "log": str(log_path),
        "started_utc": started.isoformat(),
        "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "return_code": returncode, "error": error,
        "scene_sha256": pu.sha256_file(job.experiment.scene),
    }
    if returncode != 0 or error:
        return record
    matches = OUTPUT_MARKER.findall(stdout)
    if len(matches) != 1:
        record["error"] = f"expected one profiler output marker, found {len(matches)}"
        return record
    run_dir = Path(matches[0].strip()).resolve()
    record["run_dir"] = str(run_dir)
    try:
        run = pu.load_run(run_dir)
        expected_mode = "throughput" if job.role == "throughput" else "detail"
        if pu.dotted_get(run.metadata, "profiler.mode") != expected_mode:
            raise pu.SchemaError("profiler mode does not match scheduled role")
        if bool(pu.dotted_get(run.metadata, "profiler.collect_counters")) != (
                job.role == "counter"):
            raise pu.SchemaError("counter setting does not match scheduled role")
        if not pu.frame_values(run):
            raise pu.SchemaError("no measured frames remain after warmup")
        pu.write_runner_metadata(run_dir, {
            "sequence": job.experiment.sequence,
            "experiment_id": job.experiment.experiment_id,
            "label": job.experiment.label,
            "role": job.role,
            "repetition": job.repetition,
            "baseline": job.experiment.baseline,
            "compare_to": job.experiment.compare_to,
            "claim_class": job.experiment.claim_class,
            "headline": job.experiment.headline,
            "scene_sha256": record["scene_sha256"],
            "generated_config_sha256": pu.sha256_file(config_path),
        })
    except (OSError, ValueError, pu.SchemaError) as exc:
        record["error"] = str(exc)
        return record
    record["run_json_sha256"] = pu.sha256_file(run_dir / "run.json")
    return record


def _manifest(run_root: Path, exe: Path, spec: Path, repo_root: Path,
              runner: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    status = _git(["status", "--porcelain"], repo_root)
    data = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(exe),
        "executable_sha256": pu.sha256_file(exe) if exe.is_file() else None,
        "spec": str(spec),
        "spec_sha256": pu.sha256_file(spec),
        "git_commit": _git(["rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(status),
        "platform": platform.platform(),
        "python": sys.version,
        "runner": runner,
        "jobs": records,
    }
    path = run_root / "manifest.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", help="Release path-tracer executable")
    parser.add_argument("--spec", default="scripts/experiments/project3.toml")
    parser.add_argument("--output-dir", help="override runner.output_dir")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec_path = _resolve(repo_root, args.spec)
    runner, experiments = load_spec(spec_path)
    repetitions = (args.repetitions if args.repetitions is not None
                   else int(runner.get("repetitions", 3)))
    warmup = args.warmup if args.warmup is not None else int(runner.get("warmup", 16))
    seed = args.seed if args.seed is not None else int(runner.get("seed", 5650))
    timeout = args.timeout if args.timeout is not None else int(runner.get("timeout_seconds", 900))
    if warmup < 0:
        raise SystemExit("warmup must be non-negative")
    if timeout < 1:
        raise SystemExit("timeout must be positive")
    output_setting = args.output_dir or str(runner.get("output_dir", "profiler_output"))
    output_root = _resolve(repo_root, output_setting)
    exe = _resolve(repo_root, args.exe)
    jobs = make_jobs(experiments, repetitions, seed)

    if args.dry_run:
        print(f"Validated {len(experiments)} experiments / {len(jobs)} jobs")
        for index, job in enumerate(jobs, 1):
            print(f"{index:03d}  r{job.repetition}  {job.role:10s}  "
                  f"{job.experiment.experiment_id}  {job.experiment.scene}")
        return
    if not exe.is_file():
        raise SystemExit(f"executable not found: {exe}")
    missing_scenes = [str(exp.scene) for exp in experiments if not exp.scene.is_file()]
    if missing_scenes:
        raise SystemExit("missing scene files:\n  " + "\n  ".join(missing_scenes))

    run_root = _run_root(output_root)
    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job.experiment.experiment_id} / "
              f"{job.role} / repetition {job.repetition}", flush=True)
        record = _run_job(exe, job, run_root, warmup, timeout, repo_root)
        records.append(record)
        if record.get("error"):
            print(f"  FAILED: {record['error']}", file=sys.stderr)
            if not args.keep_going:
                break
        else:
            print(f"  {record['run_dir']}")

    manifest_path = _manifest(
        run_root, exe, spec_path, repo_root,
        {**runner, "repetitions": repetitions, "warmup": warmup,
         "seed": seed, "timeout_seconds": timeout}, records)
    failures = [record for record in records if record.get("error")]
    if not failures and len(records) == len(jobs):
        from make_report import build_report
        report = build_report(run_root)
        print(f"Report: {report}")
    print(f"Manifest: {manifest_path}")
    if failures or len(records) != len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
