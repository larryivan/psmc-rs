"""Fit E1's 40 genomes with each variant of the diagnostic build, with E1's own command.

Usage: run_e1.py DIAG_BINARY RUNS [-P 4] [--only MODEL/repNN ...] [--out DIR] [--c-psmc C_BINARY]

DIAG_BINARY is psmc-rs built with diag.patch (scripts/build.sh); RUNS a runs
directory holding E1's simulated genomes (RUNS/E1/sims/<model>/repNN/genome.psmcfa:
the production's, or experiment/.production/experiment/runs on the laptop),
each checked against the sha256 its committed provenance record holds. Each
fit runs the command of E1's committed psmc-rs record (experiment/runs/E1/
provenance/fit-rs/<model>/repNN.json) verbatim but for the binary and the
paths, with the variant's environment (prec also with --mstep-iters 100000),
and writes fits/<model>/repNN.<variant>.json and .psmc (or into --out; the
HTML report is deleted). Every fit runs in a process of its own, at most -P at once.

First, bottleneck replicate 1 is fitted with no variant set; on an x86-64
CPU with AVX2 and FMA and glibc 2.34, as in the production, its JSON and
.psmc are byte-identical to the committed fit. With --c-psmc, C psmc also
fits bottleneck replicate 1 with its committed command, whose .psmc must be
byte-identical to the committed one for the same binary.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
C = Path(__file__).resolve().parent.parent
E1 = REPO / "experiment" / "runs" / "E1"
MODELS = ("bottleneck", "constant", "expansion", "zigzag")
VARIANTS = {
    "lastbin": {"PSMC_DIAG_LASTBIN": "1"},
    "hj": {"PSMC_DIAG_HJ": "1"},
    "both": {"PSMC_DIAG_LASTBIN": "1", "PSMC_DIAG_HJ": "1"},
    "prec": {"PSMC_DIAG_REL_TOL": "0"},
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command(record: Path, binary: str, runs: Path, json_out: Path, psmc_out: Path, variant: str):
    """The record's command with this binary, genome and outputs; --mstep-iters 100000 for prec."""
    key = json.loads(record.read_text())["key"]
    (genome, want), = key["inputs"].items()
    genome = runs / genome.removeprefix("<runs>/")
    if sha(genome) != want:
        raise SystemExit(f"{genome}: not the genome of {record.relative_to(REPO)}")
    cmd = list(key["command"])
    assert cmd[0] == "<repo>/target/release/psmc-rs" and cmd[3] == "25", cmd
    cmd[0], cmd[1], cmd[2] = binary, str(genome), str(json_out)
    cmd[cmd.index("--psmc-out") + 1] = str(psmc_out)
    if variant == "prec":
        cmd[cmd.index("--mstep-iters") + 1] = "100000"
    assert not any("<" in a for a in cmd), cmd
    return cmd


def fit(binary: str, runs: Path, model: str, rep: str, variant: str, out: Path) -> str:
    record = E1 / "provenance" / "fit-rs" / model / f"{rep}.json"
    js, ps = out / f"{rep}.{variant}.json", out / f"{rep}.{variant}.psmc"
    cmd = command(record, binary, runs, js, ps, variant)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PSMC_DIAG_")} | VARIANTS.get(variant, {})
    p = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    js.with_suffix(".html").unlink(missing_ok=True)
    warn = [line for line in p.stderr.splitlines() if line.lower().startswith("warning")]
    return f"{model} {rep} {variant} exit {p.returncode}" + "".join(f"\n  {w}" for w in warn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary")
    ap.add_argument("runs", type=Path)
    ap.add_argument("-P", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="MODEL/repNN genomes only")
    ap.add_argument("--out", type=Path, default=C / "fits", help="where the fits go (default: fits/)")
    ap.add_argument("--c-psmc", help="C psmc binary: fit bottleneck rep01 again and compare with the committed fit")
    a = ap.parse_args()
    binary = str(Path(a.binary).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(fit(binary, a.runs, "bottleneck", "rep01", "none", tmp))
        for ext in ("json", "psmc"):
            same = (tmp / f"rep01.none.{ext}").read_bytes() == (E1 / "fits" / "bottleneck" / f"rep01.rs.{ext}").read_bytes()
            print(f"no variant set: bottleneck rep01.{ext} {'==' if same else '!='} the committed fit")
        if a.c_psmc:
            key = json.loads((E1 / "provenance" / "fit-c" / "bottleneck" / "rep01.json").read_text())["key"]
            cmd = [a.c_psmc if x == "<tools>/psmc_c/psmc" else x for x in key["command"]]
            cmd = [str(tmp / "rep01.c.psmc") if x.startswith("<runs>/E1/fits/") else x for x in cmd]
            cmd = [str(a.runs / x.removeprefix("<runs>/")) if x.startswith("<runs>/") else x for x in cmd]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            same = (tmp / "rep01.c.psmc").read_bytes() == (E1 / "fits" / "bottleneck" / "rep01.c.psmc").read_bytes()
            print(f"C psmc again: bottleneck rep01.c.psmc {'==' if same else '!='} the committed fit")
    genomes = [(m, f"rep{r:02d}") for m in MODELS for r in range(1, 11)]
    if a.only:
        genomes = [g for g in genomes if f"{g[0]}/{g[1]}" in a.only]
        assert len(genomes) == len(a.only), a.only
    jobs = [(m, rep, v) for v in VARIANTS for m, rep in genomes]
    for m in MODELS:
        (a.out / m).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.P) as pool:
        for line in pool.map(lambda j: fit(binary, a.runs, j[0], j[1], j[2], a.out / j[0]), jobs):
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
