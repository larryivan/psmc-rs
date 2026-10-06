"""Command line of the experiment pipeline: ``python -m experiment.pipeline COMMAND``.

Run from the repository root with the experiment's environment, e.g.::

    experiment/.venv/bin/python -m experiment.pipeline plan E1
    experiment/.venv/bin/python -m experiment.pipeline start E1      # detached runner
    experiment/.venv/bin/python -m experiment.pipeline status
    experiment/.venv/bin/python -m experiment.pipeline stop

``--profile smoke`` runs the same jobs at a tiny size in experiment/.runs-smoke.
``--only REGEX`` keeps the jobs whose id matches (a shard of a run, e.g.
``--only 'E3/(sim/|run/.*/rep2$)'``); see runner.select.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from . import config, demography, paths, runner, tools
from .experiments import Context, available, jobs_for, module


def _common(p: argparse.ArgumentParser, experiments: bool = True) -> None:
    p.add_argument("--profile", default="production", choices=sorted(config.PROFILES), help="parameter profile (default production)")
    p.add_argument("--runs-dir", type=Path, help="runs directory (default: experiment/runs, or experiment/.runs-smoke for smoke)")
    if experiments:
        p.add_argument("experiments", nargs="*", metavar="EXP", help=f"experiments (default: all implemented: {' '.join(available())})")
        p.add_argument("--jobs-file", type=Path, help="run the jobs of this JSON file instead of experiments (tests, ad-hoc runs)")


def _regex(text: str) -> str:
    try:
        re.compile(text)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"not a regular expression: {exc}") from None
    return text


def _only_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--only", type=_regex, metavar="REGEX",
                   help="keep only the jobs whose id matches REGEX (re.search), e.g. 'E3/(sim/|run/.*/rep2$)'; "
                        "a kept job whose input only a dropped job makes runs if the file exists and is blocked "
                        "otherwise")


def _budget_args(p: argparse.ArgumentParser) -> None:
    b = config.Budget()
    p.add_argument("--cores", type=int, default=b.cores,
                   help=f"cores the jobs may use at once (default {b.cores}; E2's bootstraps need "
                        f"{config.E2().rs_threads}; exclusive jobs, E3's timing runs, run alone on as many cores "
                        "as threads)")
    p.add_argument("--mem-gb", type=float, default=b.mem_gb,
                   help=f"memory the jobs may use at once (default {b.mem_gb:g}; E3 needs 15)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiment.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("start", help="run every job that is not done, as a detached daemon")
    _common(p)
    _only_arg(p)
    _budget_args(p)
    p.add_argument("--foreground", action="store_true", help="run in this process instead of detaching")
    p.add_argument("--keep-awake", action=argparse.BooleanOptionalAction, default=True,
                   help="macOS: keep the machine from idle sleep while the runner lives (caffeinate -i -w PID; "
                        "default on: a sleep would stretch a timing run's wall time)")

    p = sub.add_parser("status", help="done/running/failed/pending jobs per experiment, with an ETA")
    _common(p)
    _only_arg(p)
    _budget_args(p)
    p.add_argument("--jobs", action="store_true", help="also list pending jobs and why they are not done")

    p = sub.add_parser("stop", help="stop the runner (running jobs are terminated and stay pending)")
    _common(p, experiments=False)
    p.add_argument("--drain", action="store_true", help="start no new jobs; stop when the running ones finish")
    p.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the runner to exit")

    p = sub.add_parser("plan", help="estimated CPU hours and wall time of experiments run from scratch")
    _common(p)
    _only_arg(p)
    _budget_args(p)

    p = sub.add_parser("list", help="list the jobs with their commands")
    _common(p)
    _only_arg(p)

    p = sub.add_parser("report", help="write an experiment's tables (a runner job)")
    _common(p, experiments=False)
    p.add_argument("experiment")

    p = sub.add_parser("tools", help="show the tools and their identities")
    p.add_argument("--build-rs01", action="store_true", help="build psmc-rs 0.1 (commit e50d69d) into the tools directory")
    p.add_argument("--build-c", action="store_true",
                   help="build C psmc, fq2psmcfa and splitfa in <tools>/psmc_c with $CC (default gcc, else cc), "
                        f"$PSMC_C_CFLAGS (default the Makefile's {tools.C_DEFAULT_CFLAGS!r}; on macOS arm64, where "
                        f"-O2 segfaults, {tools.C_MACOS_ARM64_CFLAGS!r}) and $LDFLAGS, and record how in its "
                        "BUILD.json; programs its BUILD.json does not describe are not replaced without --force")
    p.add_argument("--force", action="store_true",
                   help="with --build-c: replace programs built otherwise too (kept as <program>.prev)")
    sub.add_parser("check-models", help="check the demographic models against msprime and stdpopsim")
    sub.add_parser("simulate", help="simulate one genome (a runner job; see --help)", add_help=False)

    p = sub.add_parser("_daemon")  # internal: the detached runner started by `start`
    _common(p)
    _only_arg(p)
    _budget_args(p)
    sub._choices_actions = [a for a in sub._choices_actions if a.dest != "_daemon"]
    return parser


def runs_dir_of(args) -> Path:
    return (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()


def _last_meta(runs_dir: Path, profile: str) -> dict:
    """The state the running (or else the last) runner of ``runs_dir`` wrote, if it ran ``profile``."""
    try:
        state = json.loads((runner.runner_dir(runs_dir) / "state.json").read_text())
    except (OSError, ValueError):
        return {}
    return state if state.get("profile") in (None, profile) else {}


def make_graph(args, runs_dir: Path, build_tools: bool) -> runner.Graph:
    """The graph of the jobs of ``args`` (experiments or jobs file), of those ``--only`` keeps."""
    if getattr(args, "jobs_file", None):
        specs = json.loads(Path(args.jobs_file).read_text())
        jobs = [runner.Job.from_dict(d) for d in specs]
    else:
        exps = args.experiments or available()
        ctx = Context(config.get_profile(args.profile), runs_dir, build_tools=build_tools)
        jobs = jobs_for(ctx, exps)
    only = getattr(args, "only", None)
    kept = runner.select(jobs, only)
    if not kept and jobs:
        raise SystemExit(f"--only {only!r} matches none of the {len(jobs)} jobs")
    return runner.Graph(kept, runs_dir)


def cmd_start(args) -> int:
    runs_dir = runs_dir_of(args)
    pid = runner.daemon_alive(runs_dir)
    if pid is not None:
        host = runner.runner_host(runs_dir)
        where = f" on {host}" if runner.on_another_host(host) else ""
        print(f"a runner is already running for {runs_dir} (pid {pid}{where}); `stop` it first or wait", file=sys.stderr)
        return 1
    try:
        runner.sample_seconds()  # here, not in the daemon, whose log nobody reads at once
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    budget = config.Budget(cores=args.cores, mem_gb=args.mem_gb)
    for d in tools.lock_differences():
        print(f"warning: {d} (experiment/pipeline/requirements.lock); simulations may not reproduce the "
              "committed ones", file=sys.stderr)
    graph = make_graph(args, runs_dir, build_tools=True)  # fail here, not in the daemon
    exps = graph.experiments()
    meta = {
        "profile": args.profile,
        "experiments": exps,
        "budget": asdict(budget),
        "runs_dir": str(runs_dir),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jobs_file": str(args.jobs_file) if args.jobs_file else None,
        "only": args.only,
    }
    if args.foreground:
        return runner.daemon_main(lambda: make_graph(args, runs_dir, build_tools=False), runs_dir, budget, meta)
    argv = [paths.python(), "-m", "experiment.pipeline", "_daemon", "--profile", args.profile,
            "--runs-dir", str(runs_dir), "--cores", str(args.cores), "--mem-gb", str(args.mem_gb)]
    argv += ["--only", args.only] if args.only is not None else []
    argv += ["--jobs-file", str(Path(args.jobs_file).resolve())] if args.jobs_file else exps
    pid = runner.spawn_daemon(argv, runs_dir)
    if args.keep_awake and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(pid)], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"runner started (pid {pid}) for {', '.join(exps)}"
          + (f" (--only {args.only!r}: {len(graph.jobs)} jobs)" if args.only is not None else "")
          + f" with {budget.cores} cores, {budget.mem_gb:g} GB")
    print(f"  log:    {runner.runner_dir(runs_dir) / 'runner.log'}")
    print(f"  status: {paths.python()} -m experiment.pipeline status --profile {args.profile}"
          + (f" --runs-dir {runs_dir}" if args.runs_dir else ""))
    return 0


def cmd_daemon(args) -> int:
    runs_dir = runs_dir_of(args)
    budget = config.Budget(cores=args.cores, mem_gb=args.mem_gb)
    meta = {
        "profile": args.profile,
        "experiments": args.experiments or available(),
        "budget": asdict(budget),
        "runs_dir": str(runs_dir),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jobs_file": str(args.jobs_file) if args.jobs_file else None,
        "only": args.only,
    }
    return runner.daemon_main(lambda: make_graph(args, runs_dir, build_tools=True), runs_dir, budget, meta)


def cmd_status(args) -> int:
    runs_dir = runs_dir_of(args)
    # Without experiments named: those of the running runner, or else of the last one,
    # and its --only unless one is given.
    meta = _last_meta(runs_dir, args.profile)
    inherited = False
    if not args.experiments and not args.jobs_file and meta:
        args.experiments = meta.get("experiments") or []
        if meta.get("jobs_file"):
            args.jobs_file = Path(meta["jobs_file"])
        if args.only is None:
            args.only = meta.get("only")
            inherited = args.only is not None
    budget = config.Budget(**meta["budget"]) if meta.get("budget") else config.Budget(args.cores, args.mem_gb)
    graph = make_graph(args, runs_dir, build_tools=False)
    pid = runner.daemon_alive(runs_dir)
    statuses = runner.job_statuses(graph, runs_dir, daemon_pid=pid)
    print(runner.format_status(graph, statuses, budget, runs_dir, pid, verbose=args.jobs))
    if args.only:  # a merged runs directory's state holds one shard's filter
        print(f"(only the {len(graph.jobs)} jobs that --only {args.only!r}"
              + (" of the last runner" if inherited else "") + " keeps; --only '' shows all)")
    return 0


def cmd_stop(args) -> int:
    runs_dir = runs_dir_of(args)
    try:
        pid = runner.stop_daemon(runs_dir, drain=args.drain, wait=args.wait)
    except runner.ElsewhereError as exc:
        print(f"not stopped: {exc}", file=sys.stderr)
        return 1
    if pid is None:
        print("no runner is running")
        return 0
    if args.drain:
        print(f"runner {pid} drains: it starts no new jobs and exits when the running ones finish")
        return 0
    if runner.daemon_alive(runs_dir) is not None:
        print(f"runner {pid} has not exited after {args.wait:g} s", file=sys.stderr)
        return 1
    print(f"runner {pid} stopped; jobs it was running stay pending and run again at the next start")
    return 0


def cmd_plan(args) -> int:
    runs_dir = runs_dir_of(args)
    budget = config.Budget(cores=args.cores, mem_gb=args.mem_gb)
    graph = make_graph(args, runs_dir, build_tools=False)
    est = runner.plan(graph, budget)
    print(f"profile {args.profile}, budget {budget.cores} cores, {budget.mem_gb:g} GB (estimates from config.Costs)")
    print(f"{'experiment':<11}{'jobs':>6}{'CPU h':>9}{'wall h (alone)':>16}{'finish h (all)':>16}")
    for exp, e in est.items():
        print(f"{exp:<11}{e['jobs']:>6}{e['cpu_hours']:>9.2f}{e['wall_hours_alone']:>16.2f}{e['finish_hours']:>16.2f}")
    kinds = {}
    for j in graph.jobs.values():
        kind = j.name.split("/", 1)[0]
        k = kinds.setdefault((j.experiment, kind), [0, 0.0, 0.0, 0.0])
        k[0] += 1
        k[1] += j.est_seconds * j.cores
        k[2] = max(k[2], j.mem_gb)
        k[3] = max(k[3], j.est_seconds)
    print(f"\n{'job kind':<20}{'jobs':>6}{'CPU h':>9}{'max GB':>8}{'max min':>9}")
    for (exp, kind), (n, cpu, mem, longest) in kinds.items():
        print(f"{exp + '/' + kind:<20}{n:>6}{cpu / 3600:>9.2f}{mem:>8.2f}{longest / 60:>9.1f}")
    return 0


def cmd_list(args) -> int:
    runs_dir = runs_dir_of(args)
    graph = make_graph(args, runs_dir, build_tools=False)
    for jid in graph.order:
        j = graph.jobs[jid]
        flags = f"{j.cores} core(s), {j.mem_gb:g} GB, ~{runner.fmt_duration(j.est_seconds)}" + (", exclusive" if j.exclusive else "")
        print(f"{jid}  [{flags}]\n    {' '.join(paths.portable(c, runs_dir) for c in j.command)}")
    return 0


def cmd_report(args) -> int:
    runs_dir = runs_dir_of(args)
    ctx = Context(config.get_profile(args.profile), runs_dir)
    written = module(args.experiment).report(ctx)
    for key, path in written.items():
        print(f"{key}: {path}")
    return 0


def cmd_tools(args) -> int:
    if args.build_c:
        try:
            tools.build_c(force=args.force)
        except (tools.ToolError, OSError) as exc:
            print(f"building C psmc failed: {exc}", file=sys.stderr)
            return 1
    found: List[tools.Tool] = []
    problems: List[str] = []
    attempts = [
        ("psmc-rs 0.2", lambda: tools.psmc_rs_02(build=True)),
        ("C psmc", tools.c_psmc),
        ("psmc-rs 0.1", lambda: tools.psmc_rs_01(build=args.build_rs01)),
        *[(name, (lambda n=name: tools.utility(n))) for name in tools.UTILITIES],
    ]
    for label, get in attempts:
        try:
            found.append(get())
        except (tools.ToolError, OSError) as exc:
            problems.append(f"{label}: {exc}")
    print(f"tools directory: {paths.tools_dir()}")
    print(tools.describe(found))
    for k, v in tools.htslib_versions().items():
        print(f"{k:24s} {v}")
    for p in problems:
        print(f"missing  {p}")
    for d in tools.lock_differences():
        print(f"warning  {d}: simulations may not reproduce the committed ones")
    # A missing or unbuildable tool fails the command, so that a build script
    # can tell (each one is printed above).
    return 1 if problems else 0


def cmd_check_models(args) -> int:
    for name in config.MODEL_ORDER:
        res = demography.compare_with_debugger(config.MODELS[name])
        print(f"{name}: {res['epochs']} epochs agree with msprime's DemographyDebugger (max rel diff {res['max_rel_diff']:.2g})")
    res = demography.check_zigzag_against_stdpopsim()
    print(f"zigzag: equal to stdpopsim {res['stdpopsim_version']} {res['stdpopsim_model']} "
          f"({res['epochs']} epochs, max rel diff {res['max_rel_diff']:.2g}, coalescence probabilities "
          f"max abs diff {res['max_abs_diff_coalescence_probability']:.2g})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["simulate"]:
        from . import simulate

        return simulate.main(argv[1:])
    args = build_parser().parse_args(argv)
    handlers = {
        "start": cmd_start, "_daemon": cmd_daemon, "status": cmd_status, "stop": cmd_stop, "plan": cmd_plan,
        "list": cmd_list, "report": cmd_report, "tools": cmd_tools, "check-models": cmd_check_models,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
