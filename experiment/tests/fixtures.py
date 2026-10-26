"""What the experiment tests share: the repository's paths, stand-in tools, and small psmc-rs / C psmc outputs.

Importing it puts the repository root on ``sys.path`` (for ``experiment.pipeline``). It stays light to
import: the pipeline is imported only where needed.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"  # C psmc's output, 5 iterations


def has_modules(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def has_font(name: str) -> bool:
    """Whether matplotlib finds the font ``name`` itself, not a fallback."""
    try:
        from matplotlib import font_manager
        font_manager.findfont(name, fallback_to_default=False)
    except (ImportError, ValueError):
        return False
    return True


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quiet() -> contextlib.ExitStack:
    """Silence what the code under test prints, on stdout and stderr."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def quick_runner() -> contextlib.ExitStack:
    """The runner in this process, for jobs of a second or less: it looks for finished jobs every 0.05 s (not
    0.25 s), and samples the busiest other processes before an exclusive job for 0.05 s (not 0.5 s)."""
    from unittest import mock

    from experiment.pipeline import runner

    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(runner, "POLL_SECONDS", 0.05))
    stack.enter_context(mock.patch.object(runner, "busiest_processes",
                                          functools.partial(runner.busiest_processes, interval=0.05)))
    return stack


# Stand-ins for the tools a Context identifies (Context._tools): their identities enter the jobs, but no job runs.
_FAKE_TOOLS = {
    "c": ("c-psmc", "/x/psmc", {"version": "0.6.5", "sha256": "0" * 64}),
    "rs02": ("psmc-rs-0.2", "/x/psmc-rs", {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40}),
    "fq2psmcfa": ("fq2psmcfa", "/x/fq2psmcfa", {"sha256": "1" * 64}),
    "generate_multihetsep.py": ("generate_multihetsep.py", "/x/gm.py", {"sha256": "2" * 64}),
}


def fake_tools(*keys: str) -> dict:
    """The stand-in tools of ``keys`` (default: all of them), as a new dict for Context._tools."""
    from experiment.pipeline import tools

    return {k: tools.Tool(name, Path(path), dict(info)) for k, (name, path, info) in _FAKE_TOOLS.items()
            if k in (keys or _FAKE_TOOLS)}


def fake_ctx(profile, runs: Path, *keys: str, **kw):
    """A Context of ``profile`` in ``runs`` with the stand-in tools of ``keys`` (default: all)."""
    from experiment.pipeline.experiments import Context

    return Context(profile, runs, _tools=fake_tools(*keys), **kw)


def e7_sample(sid: str, pop: str = "YRI", sup: str = "AFR", default: bool = False, split: bool = False,
              boot: bool = False, giab: bool = False):
    """A sample of E7's sample table with these roles (a kg1000.Sample; a cohort sample unless ``default``)."""
    from experiment.pipeline import kg1000

    return kg1000.Sample(sample_id=sid, population=pop, superpopulation=sup, sex="female", is_parent=False,
                         admixed=False, raw_bases=93_000_000_000, raw_depth=30.0, default=default,
                         selection="drawn" if default else "cohort", split_check=split, bootstrap=boot, giab=giab)


def stage_e7(data, samples: Iterable) -> None:
    """E7's data stage finished in ``data`` (a kg1000.Layout): the sample table of ``samples``, a chunk plan of chr1
    and chr2 and MANIFEST.json, so E7 declares its jobs; none of the inputs its fits read."""
    from experiment.pipeline import kg1000

    kg1000.write_tsv(data.samples, [s.row() for s in samples], kg1000.SAMPLE_COLUMNS)
    kg1000.write_tsv(data.chunks, [{"chrom": "chr1", "chunk": 0, "start": 1, "end": 248_956_422},
                                   {"chrom": "chr2", "chunk": 0, "start": 1, "end": 242_193_529}],
                     ("chrom", "chunk", "start", "end"))
    kg1000.write_json(data.manifest, {"manifest": kg1000.MANIFEST_ID})


def option(command: Sequence, name: str) -> Optional[str]:
    """The value ``command`` gives its option ``name`` (None without the option)."""
    command = [str(c) for c in command]
    return command[command.index(name) + 1] if name in command else None


@functools.lru_cache(maxsize=None)
def _c_ref():
    from experiment.pipeline import results

    return results.parse_psmc(C_REF)


def c_ref_json(lam_scale: float = 1.0, **fields) -> dict:
    """psmc-rs's JSON of C psmc's reference fit (its last round), the lambdas times ``lam_scale``, and ``fields``."""
    ref = _c_ref()
    pa = ref.last().pa
    return {"theta": pa["theta"], "rho": pa["rho"], "t_max": pa["t_max"], "n_steps": ref.header["n"],
            "pattern": ref.header["pattern"], "lam": [x * lam_scale for x in pa["lambdas"]], **fields}


def c_ref_loglikes(final: float = -25000.0) -> List[float]:
    """psmc-rs's trace of the reference's 5 iterations: C psmc's log-likelihoods, then ``final`` (of the last model,
    which C psmc never evaluates)."""
    ll = _c_ref().loglikes()
    return [ll[i] for i in range(5)] + [final]


@functools.lru_cache(maxsize=None)
def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def _repo_path(path) -> str:
    return Path(path).relative_to(ROOT).as_posix() if Path(path).is_absolute() else str(path)


def assert_ignore_rules(test, kept: Iterable = (), ignored: Iterable = ()) -> None:
    """Git keeps every path of ``kept`` and ignores every one of ``ignored`` (repository-relative or absolute).

    One ``git check-ignore`` for all, in an empty repository holding only this one's ``.gitignore`` files: by
    the rules alone, whatever is on disk (a production links E7's inputs, and git checks no path beyond a link).
    """
    kept, ignored = [_repo_path(p) for p in kept], [_repo_path(p) for p in ignored]
    rules = subprocess.run(["git", "ls-files", "-z", "--", ".gitignore", "*/.gitignore"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split("\0")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        for rule in filter(None, rules):
            (Path(tmp) / rule).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rule, Path(tmp) / rule)
        res = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"], input="\0".join(kept + ignored),
                             cwd=tmp, capture_output=True, text=True)
    test.assertIn(res.returncode, (0, 1), res.stderr)  # 1: none is ignored
    hit = set(res.stdout.split("\0"))
    with test.subTest("kept"):
        test.assertEqual([p for p in kept if p in hit], [], "ignored, but git must keep them")
    with test.subTest("ignored"):
        test.assertEqual([p for p in ignored if p not in hit], [], "kept, but git must ignore them")
