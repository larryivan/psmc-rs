"""The experiment pipeline's command line.

Run it from the repository root with the experiment's environment:

    PY=experiment/.venv/bin/python
    $PY -m experiment.pipeline plan E1      # what E1's jobs cost on the budget
    $PY -m experiment.pipeline start E1     # run what is not done, in a detached runner
    $PY -m experiment.pipeline status       # done, running, failed and blocked jobs; ETA
    $PY -m experiment.pipeline stop         # start again resumes

--profile smoke runs the same jobs at a small size in experiment/.runs-smoke;
--only REGEX keeps the jobs whose id matches (a shard of a run). COMMAND
--help describes a command, experiment/pipeline/README.md the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config, demography, paths, provenance, runner, tools
from .experiments import Context, available, jobs_for, module


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="production", choices=sorted(config.PROFILES),
                   help="production (the manuscript's; the default), smoke (the same jobs at a small size) or tiny "
                        "(the tests')")
    p.add_argument("--runs-dir", type=Path, metavar="DIR",
                   help="runs directory (default: $PSMC_EXP_RUNS or experiment/runs; for smoke experiment/.runs-smoke, "
                        "for tiny experiment/.runs-smoke/tiny)")


def _regex(text: str) -> str:
    try:
        re.compile(text)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"not a regular expression: {exc}") from None
    return text


def _job_args(p: argparse.ArgumentParser, budget: bool = True, last: str = "") -> None:
    """The arguments of the commands that make the jobs' graph, and with ``budget`` the budget.

    ``last`` heads the defaults of the command that takes what it is not given from the last runner (status).
    """
    _common(p)
    p.add_argument("experiments", nargs="*", metavar="EXP", type=str.upper, choices=available(),
                   help=f"experiments (default: {last}all of them, {' '.join(available())})")
    p.add_argument("--jobs-file", type=Path, metavar="FILE",
                   help="the jobs of this JSON file instead of the experiments' (tests, ad-hoc runs)")
    p.add_argument("--only", type=_regex, metavar="REGEX",
                   help="keep only the jobs whose id matches REGEX (re.search), e.g. 'E3/(sim/|run/.*/rep2$)'; "
                        "a kept job whose input only a dropped job makes runs if the file exists and is blocked "
                        "otherwise")
    if not budget:
        return
    b = config.Budget()
    p.add_argument("--cores", type=int, metavar="N",
                   help=f"cores the jobs may use at once (default: {last}{b.cores}; E2's bootstraps need "
                        f"{config.E2().rs_threads}; exclusive jobs, E3's timing runs, run alone on as many cores as "
                        "threads)")
    p.add_argument("--mem-gb", type=float, metavar="GB",
                   help=f"memory the jobs may use at once (default: {last}{b.mem_gb:g})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiment.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, summary: str, details: str = "", **kw) -> argparse.ArgumentParser:
        description = f"{summary[0].upper()}{summary[1:]}." + (f" {details}" if details else "")
        return sub.add_parser(name, help=summary, description=description, **kw)

    p = command("start", "run every job that is not done, in a detached runner",
                "A job is done when its outputs exist with a matching provenance record; a failed one runs again. "
                "status follows the runner, stop stops it, and start again resumes.")
    _job_args(p)
    p.add_argument("--foreground", action="store_true", help="run in this process instead of detaching")
    p.add_argument("--keep-awake", action=argparse.BooleanOptionalAction, default=True,
                   help="macOS: keep the machine from idle sleep while the runner lives (caffeinate -i -w PID; "
                        "default on: a sleep would stretch a timing run's wall time)")

    p = command("status", "done, running, failed, blocked and pending jobs per experiment, with an ETA",
                "What is not given (experiments or jobs file, --only, --cores, --mem-gb) is the running runner's, or "
                "else the last one's. A blocked job waits for what no runner does: an input that no job makes, more "
                "cores or memory, or a blocked job it needs.")
    _job_args(p, last="the last runner's, else ")
    p.add_argument("--jobs", action="store_true",
                   help="list every job that is not done, and why (without it: at most "
                        f"{runner.STATUS_LINES} running, {runner.STATUS_LINES} failed and {runner.STATUS_LINES} "
                        "blocked jobs of each experiment)")

    p = command("stop", "stop the runner: the running jobs are terminated and stay pending",
                "--drain lets them finish instead. start again resumes.")
    _common(p)
    p.add_argument("--drain", action="store_true", help="start no new jobs; stop when the running ones finish")
    p.add_argument("--wait", type=float, default=120.0, metavar="SECONDS",
                   help="how long to wait for the runner to exit (default 120)")

    p = command("plan", "estimated CPU hours and wall time of the experiments run from scratch on the budget",
                "From the cost model of config.py (Costs); the jobs that can never start on the budget are listed.")
    _job_args(p)

    p = command("list", "the jobs with their resources and commands")
    _job_args(p, budget=False)

    p = command("report", "write an experiment's tables from its outputs (a runner job)")
    _common(p)
    p.add_argument("experiment", metavar="EXP", type=str.upper, choices=available(),
                   help="E1 to E7, or FIG (the figures)")

    p = command("tools", "show the tools the jobs run and their identities", "Exits 1 if one is missing.")
    p.add_argument("--build-c", action="store_true",
                   help="build C psmc, fq2psmcfa and splitfa in <tools>/psmc_c with $CC (default gcc, else cc), "
                        f"$PSMC_C_CFLAGS (default the Makefile's {tools.C_DEFAULT_CFLAGS!r}; on macOS arm64, where "
                        f"-O2 segfaults, {tools.C_MACOS_ARM64_CFLAGS!r}) and $LDFLAGS, and record how in its "
                        "BUILD.json; programs its BUILD.json does not describe are not replaced without --force")
    p.add_argument("--force", action="store_true",
                   help="with --build-c: replace programs built otherwise too (kept as <program>.prev)")
    command("check-models", "check the demographic models against msprime and stdpopsim")
    command("simulate", "simulate one genome (a runner job; simulate --help lists its options)", add_help=False)

    _job_args(sub.add_parser("_daemon"))  # the detached runner that start starts; no help: not listed
    return parser


def runs_dir_of(args) -> Path:
    return (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()


def _budget(args, last: Optional[Dict] = None) -> config.Budget:
    """``--cores`` and ``--mem-gb``, else those of ``last`` (a runner's recorded budget), else the defaults."""
    default = config.Budget(**last) if last else config.Budget()
    return config.Budget(cores=default.cores if args.cores is None else args.cores,
                         mem_gb=default.mem_gb if args.mem_gb is None else args.mem_gb)


def _meta(args, runs_dir: Path, budget: config.Budget, experiments: List[str]) -> dict:
    """How a runner was started, which its state records (``status`` reads it back)."""
    return {
        "profile": args.profile,
        "experiments": experiments,
        "budget": asdict(budget),
        "runs_dir": str(runs_dir),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jobs_file": str(args.jobs_file) if args.jobs_file else None,
        "only": args.only,
    }


def make_graph(args, runs_dir: Path, build_tools: bool) -> runner.Graph:
    """The graph of the jobs of ``args`` (experiments or jobs file), of those ``--only`` keeps."""
    if args.jobs_file:
        try:
            jobs = [runner.Job.from_dict(d) for d in json.loads(Path(args.jobs_file).read_text())]
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"--jobs-file {args.jobs_file}: {exc}") from None
    else:
        ctx = Context(config.get_profile(args.profile), runs_dir, build_tools=build_tools)
        jobs = jobs_for(ctx, args.experiments or available())
    kept = runner.select(jobs, args.only)
    if not kept and jobs:
        raise SystemExit(f"--only {args.only!r} matches none of the {len(jobs)} jobs")
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
    budget = _budget(args)
    for d in tools.lock_differences():
        print(f"warning: {d} (experiment/pipeline/requirements.lock); simulations may not reproduce the "
              "committed ones", file=sys.stderr)
    graph = make_graph(args, runs_dir, build_tools=True)  # fail here, not in the daemon
    exps = graph.experiments()
    if args.foreground:
        return runner.daemon_main(lambda: make_graph(args, runs_dir, build_tools=False), runs_dir, budget,
                                  _meta(args, runs_dir, budget, exps))
    argv = [paths.python(), "-m", "experiment.pipeline", "_daemon", "--profile", args.profile,
            "--runs-dir", str(runs_dir), "--cores", str(budget.cores), "--mem-gb", str(budget.mem_gb)]
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
    budget = _budget(args)
    return runner.daemon_main(lambda: make_graph(args, runs_dir, build_tools=True), runs_dir, budget,
                              _meta(args, runs_dir, budget, args.experiments or available()))


def cmd_status(args) -> int:
    runs_dir = runs_dir_of(args)
    # What is not given is the running runner's, or else the last one's, if it ran this profile.
    last = runner.read_state(runs_dir)
    if last.get("profile") not in (None, args.profile):
        last = {}
    inherited = False
    if not args.experiments and not args.jobs_file and last:
        args.experiments = last.get("experiments") or []
        if last.get("jobs_file"):
            args.jobs_file = Path(last["jobs_file"])
        if args.only is None:
            args.only = last.get("only")
            inherited = args.only is not None
    budget = _budget(args, last.get("budget"))
    graph = make_graph(args, runs_dir, build_tools=False)
    pid = runner.daemon_alive(runs_dir)
    statuses = runner.job_statuses(graph, runs_dir, daemon_pid=pid, budget=budget)
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


def _hours(h: float) -> str:
    return f"{h:.2f}" if math.isfinite(h) else "never"


def cmd_plan(args) -> int:
    budget = _budget(args)
    graph = make_graph(args, runs_dir_of(args), build_tools=False)
    print(f"profile {args.profile}, budget {budget.cores} cores, {budget.mem_gb:g} GB (estimates from config.Costs)")
    print(f"{'experiment':<11}{'jobs':>6}{'CPU h':>9}{'wall h (alone)':>16}{'finish h (all)':>16}")
    for exp, e in runner.plan(graph, budget).items():
        print(f"{exp:<11}{e['jobs']:>6}{e['cpu_hours']:>9.2f}{_hours(e['wall_hours_alone']):>16}"
              f"{_hours(e['finish_hours']):>16}")
    kinds: Dict[str, List[runner.Job]] = {}
    for j in graph.jobs.values():
        kinds.setdefault(f"{j.experiment}/{j.name.split('/', 1)[0]}", []).append(j)
    width = max([20, *(len(k) + 2 for k in kinds)])
    print(f"\n{'job kind':<{width}}{'jobs':>6}{'CPU h':>9}{'max GB':>8}{'max min':>9}")
    for kind, jobs in kinds.items():
        print(f"{kind:<{width}}{len(jobs):>6}{sum(j.est_seconds * j.cores for j in jobs) / 3600:>9.2f}"
              f"{max(j.mem_gb for j in jobs):>8.2f}{max(j.est_seconds for j in jobs) / 60:>9.1f}")
    never = {kind: [why for j in jobs if (why := runner.fits_budget(j, budget))] for kind, jobs in kinds.items()}
    never = {kind: whys for kind, whys in never.items() if whys}
    if never:
        print("\nnever start on this budget, nor do the jobs that need them (see --cores and --mem-gb):")
        for kind, whys in never.items():
            print(f"  {kind} ({len(whys)} job{'s' if len(whys) > 1 else ''}): {'; '.join(dict.fromkeys(whys))}")
    return 0


def cmd_list(args) -> int:
    runs_dir = runs_dir_of(args)
    graph = make_graph(args, runs_dir, build_tools=False)
    for jid in graph.order:
        j = graph.jobs[jid]
        flags = f"{j.cores} core(s), {j.mem_gb:g} GB, ~{runner.fmt_duration(j.est_seconds)}" + (", exclusive" if j.exclusive else "")
        print(f"{jid}  [{flags}]\n    {' '.join(provenance.portable(j.command, runs_dir))}")
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
    try:
        return handlers[args.command](args)
    except (tools.ToolError, runner.GraphError) as exc:  # a missing tool, a bad graph: the message says it all
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
