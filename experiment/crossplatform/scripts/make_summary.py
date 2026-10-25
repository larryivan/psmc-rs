"""Write summary.md from summary.json, inputs.json, reports.json, toolchain.txt, the CSVs and the run records.

Run after compare.py and reports.py: experiment/.venv/bin/python -I -B make_summary.py
Every number in summary.md comes from those files (or from the Ibex records and E3's
environment table in experiment/runs); the prose around them is fixed here.
"""
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent
IBX = REPO / "experiment/runs"
S = json.loads((X / "summary.json").read_text())
I = json.loads((X / "inputs.json").read_text())
R = json.loads((X / "reports.json").read_text())
JOBS = json.loads((X / "jobs.json").read_text())
MODELS = ("constant", "bottleneck", "expansion", "zigzag")
# The production compared (experiment/runs as committed): where its psmc-rs fits ran.
PRODUCTION_COMMIT, PRODUCTION_RUN_AT = "cbab860c", "4b2c636f"


def rows(name):
    p = X / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x, d=2):
    if x is None or x == "":
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    return f"{x:.{d - 1}e}"


def rng(st):
    return "–" if st is None else f"{f(st['min'])} / {f(st['median'])} / {f(st['max'])}"


def tiers(rr, k="tier"):
    cc = Counter(r[k] for r in rr)
    return ", ".join(f"{t} {cc[t]}" for t in ("A", "B", "C", "fail") if cc.get(t)) or "–"


def yes(v):
    return v in (True, "True")


def name(r):
    return f"{r['model']} rep{int(r['replicate']):02d}" if r["exp"] == "E1" else r["model"]


def lam(s):
    """lam3 -> λ3, t_max -> t_max."""
    return "λ" + s[3:] if s.startswith("lam") else s


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def counts(c):
    return ", ".join(f"{lam(k)} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


rs = rows("rs_laptop_vs_ibex.csv")
rs1 = [r for r in rs if r["exp"] == "E1"]
na = next(r for r in rs if r["exp"] == "E2")
c = rows("c_laptop_vs_ibex.csv")
rsc = rows("rs_vs_c_ibex.csv")
side = rows("side_by_side_E1.csv")
r199 = rows("rs_laptop_rustc1.99_vs_ibex.csv")
r199_1 = [r for r in r199 if r["exp"] == "E1"]
ctl199 = rows("control_rustc1.99_vs_rustc1.97_laptop.csv")
rep = rows("control_repeat_laptop.csv")
prm = rows("params_rel_per_fit.csv")
tool = (X / "toolchain.txt").read_text()
env = {r["key"]: r["value"] for r in csv.DictReader(open(IBX / "E3/tables/E3_environment.csv"))}
records = [json.loads(p.read_text()) for p in sorted((X / "records").glob("*.json"))]
run_dates = sorted({rec["start"][:10] for d in ("records", "records-rustc1.99", "records-repeat")
                    for rec in (json.loads(p.read_text()) for p in sorted((X / d).glob("*.json")))})

# toolchain.txt
blocks = dict(re.findall(r"== (.+?)\n(.*?)(?=\n== |\Z)", tool, re.S))
lap = blocks["laptop"]
cpu = lap.splitlines()[0]
cores = re.search(r"cores: (\d+) \(P (\d+), E (\d+)\); mem (\d+) GB", lap)
macos = re.search(r"ProductVersion:\s+(\S+)", lap).group(1)
macbuild = re.search(r"BuildVersion:\s+(\S+)", lap).group(1)
darwin = re.search(r"^Darwin (\S+)", lap, re.M).group(1)
ids = dict(re.findall(r"(src|Cargo\.toml|Cargo\.lock)=([0-9a-f]+)", lap))
dirty = int(re.search(r"dirty_files=(\d+)", lap).group(1))
head = re.search(r"git HEAD ([0-9a-f]+)", lap).group(1)
cargo_cfg = re.search(r"\.cargo/config\.toml: (.+)", lap).group(1)
builds = {}
for k, v in blocks.items():
    m = re.match(r"build <xplat>/(\S+): (.+)", k)
    if m:
        builds[m.group(1)] = {
            "cargo": m.group(2),
            "rustc": re.search(r"^rustc (.+)$", v, re.M).group(1),
            "llvm": re.search(r"LLVM version: (\S+)", v).group(1),
            "host": re.search(r"host: (\S+)", v).group(1),
            "sha": re.search(r"^([0-9a-f]{64})  ", v, re.M).group(1),
            "digest": re.search(r"^sha256 without LC_UUID and code signature: ([0-9a-f]{64})  ", v, re.M).group(1),
            "version": re.search(r"^(psmc-rs \S+)$", v, re.M).group(1),
            "neon": 'target_feature="neon"' in v,
        }
main_b, ctl_b = builds["bin/psmc-rs"], builds["bin/psmc-rs-rustc1.99"]
cblock = blocks["laptop C psmc (frozen .production run)"]
c_sha = re.search(r"^([0-9a-f]{64})  ", cblock, re.M).group(1)
text_sha = dict(re.findall(r"^(\S+) __text sha256: (\S+)", cblock, re.M))
n_fma = int(re.search(r"fused multiply-add instructions in psmc: (\d+)", cblock).group(1))
producer = re.search(r"DWARF producer: (.+)", cblock).group(1)
assert text_sha["psmc"] == text_sha["psmc_O"] == text_sha["psmc_O1"] != text_sha["psmc_O0"], text_sha

# the Ibex records: tools, hosts and jobs
def ibex_records(kind):
    out = [json.loads((IBX / f"E1/provenance/{kind}/{m}/rep{r:02d}.json").read_text()) for m in MODELS for r in range(1, 11)]
    if kind == "fit-rs":
        out.append(json.loads((IBX / "E2/provenance/fit-rs/NA12878.json").read_text()))
    return out


ib_rs, ib_c = ibex_records("fit-rs"), ibex_records("fit-c")
rs_tool = {json.dumps(x["key"]["tools"]["psmc-rs-0.2"], sort_keys=True) for x in ib_rs}
assert len(rs_tool) == 1
rs_tool = json.loads(rs_tool.pop())
ib_ids = dict(re.findall(r"(src|Cargo\.toml|Cargo\.lock)=([0-9a-f]+)", rs_tool["source_tree"]))
ib_bin = {v for x in ib_rs for v in x["run"]["executables"].values()}
assert len(ib_bin) == 1
ib_bin = ib_bin.pop()
c_tool = {json.dumps(x["key"]["tools"]["c-psmc"], sort_keys=True) for x in ib_c}
assert len(c_tool) == 1
c_tool = json.loads(c_tool.pop())
lap_c_tool = {r["laptop_c_sha256"] for r in c}
assert lap_c_tool == {c_sha}, (lap_c_tool, c_sha)


def hosts(recs):
    cc = Counter((x["run"]["host"], x["run"].get("slurm_job_id")) for x in recs)
    return "; ".join(f"{h}, SLURM job {j}, {plural(n, 'fit')}" for (h, j), n in cc.items())


rs_hosts, c_hosts = hosts(ib_rs), hosts(ib_c)
c_days = sorted({x["run"]["start"][:10] for x in ib_c})
lap_c_days = sorted({json.loads(p.read_text())["run"]["start"][:10]
                     for p in (REPO / "experiment/.production/experiment/runs/E1/provenance/fit-c").glob("*/*.json")})
assert main_b["sha"] == {r["laptop_binary_sha256"] for r in rs}.pop() and len({r["laptop_binary_sha256"] for r in rs}) == 1
assert all(r["laptop_threads"] == r["ibex_threads"] == "1" for r in rs)

c_top = max(c, key=lambda r: float(r["param_max_rel"]))
c_rest = [r for r in c if r is not c_top]
fails = [r for r in rs1 if r["tier"] == "fail"]
nb = Counter(r["tier"] for r in rs1)
RA = S["runaway"]


def mx(rr, k):
    return max(float(r[k]) for r in rr)


def med(rr, k):
    return statistics.median(float(r[k]) for r in rr)


def ratio(k):
    v = [float(s[k]) for s in side]
    return f"{f(min(v))} / {f(statistics.median(v))} / {f(max(v))}"


def rq(k, q):
    v = [float(x[k]) for x in side]
    x = min(v) if q == "min" else statistics.median(v)
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


out = []
w = out.append
w("# Cross-platform comparison: laptop vs Ibex (data for Supplementary Note 1, section 9)\n")
w(f"Computed on {' and '.join(run_dates)} by the scripts of `scripts/` (`make_jobs.py`, `run_job.py`, `toolchain.py`, `inputs.py`, `compare.py`, `reports.py`, `make_summary.py`; README.md gives the order). The tiers of section 5 are those `scripts/tiers_2026-10-07.py` defined on 2026-10-07, before these runs.\n")
w(f"The Ibex fits compared are the production's, committed at {PRODUCTION_COMMIT}: its psmc-rs fits ran at {PRODUCTION_RUN_AT} with a binary built from `src` {ib_ids['src'][:7]}, the sources the laptop built; its C psmc fits come from earlier jobs of the same C psmc binary, which did not change.\n")

w("## Headline\n")
e1 = S["rs"]["E1"]
n_rs, n_bytes = S["rs"]["n"], S["rs"]["json_bytes_identical"]
w(f"- **Inputs identical:** the laptop's {I['genomes']} E1 genomes are byte-identical to Ibex's (psmcfa, VCF.gz and index: {I['n_psmcfa_eq_ibex_truth']}/40, {I['n_vcf_gz_eq_ibex_truth']}/40, {I['n_tbi_eq_ibex_sim_prov']}/40; {I['n_chromosomes_all_counts_match']}/{I['chromosomes']} per-chromosome checksums and counts); NA12878's input too.")
w(f"- **psmc-rs, same sources, laptop (arm64 NEON, macOS) vs Ibex (x86-64 AVX2+FMA, Linux): {n_bytes}/{n_rs} outputs byte-identical.** Median (max) over the 40 E1 fits: θ {f(e1['theta_rel']['median'])} ({f(e1['theta_rel']['max'])}) relative, any parameter {f(e1['param_max_rel']['median'])} ({f(e1['param_max_rel']['max'])}), log10 Ne per interval {f(e1['dlog10ne_max_all_intervals']['median'])} ({f(e1['dlog10ne_max_all_intervals']['max'])}), window RMSE {f(e1['dlog10ne_rmse_window']['median'])} ({f(e1['dlog10ne_rmse_window']['max'])}), final log-likelihood {f(e1['loglike_rel_final']['median'])} ({f(e1['loglike_rel_final']['max'])}). NA12878: parameters {f(na['param_max_rel'])}, log10 Ne {f(na['dlog10ne_max_all_intervals'])}, window RMSE {f(na['dlog10ne_rmse_window'])}, log-likelihood {f(na['loglike_rel_final'])}.")
w(f"- **Pre-registered tiers, psmc-rs:** B {nb['B']}, C {nb['C']}, **fail {nb['fail']}** of 40 (NA12878: {na['tier']}). **C psmc (laptop Apple clang build vs Ibex gcc -O3 build): {tiers(c)} of 40.**")
w(f"- **Side by side (window RMSE, per replicate):** psmc-rs's cross-platform difference is {rq('ratio_rmse_rsvsc_over_rs_xplat', 'med')} times (median; at least {rq('ratio_rmse_rsvsc_over_rs_xplat', 'min')}) smaller than its difference from C psmc, and {rq('ratio_rmse_c_xplat_over_rs_xplat', 'med')} times (median; at least {rq('ratio_rmse_c_xplat_over_rs_xplat', 'min')}) smaller than C psmc's own cross-build difference, which is in turn {rq('ratio_rmse_rsvsc_over_c_xplat', 'med')} times smaller than psmc-rs vs C (median; C's cross-build RMSE is the larger in {len(side) - S['side_by_side_counts']['c_xplat_rmse_below_rs_vs_c']}/{len(side)}).")
ctl_same = sum(yes(r["json_bytes_identical"]) and yes(r["psmc_bytes_identical"]) for r in ctl199)
rep_same = sum(yes(r["json_bytes_identical"]) and yes(r["psmc_bytes_identical"]) for r in rep)
w(f"- **Controls:** the same sources built on the laptop by Ibex's rustc ({ctl_b['rustc'].split()[0]}) give output byte-identical to the rustc {main_b['rustc'].split()[0]} build's in {ctl_same}/{len(ctl199)} fits, and one binary run twice gives identical bytes ({rep_same}/{len(rep)})"
  + (": the differences are the platform's (architecture, kernels, math library), not the compiler version's or run-to-run noise." if ctl_same == len(ctl199) and rep_same == len(rep) else "."))
w(f"- **The analysis code:** E1, E2, E4 and E5's reports, run on the laptop on the committed fits with the code that wrote the committed tables on Ibex, write {R['tables_bytes_identical']} of the {R['tables']} tables byte for byte; the others agree to a relative {f(R['max_rel_ge_1e-3'])} in every value of magnitude 1e-3 or more (section 4.6).\n")

w("## 1. Machines and toolchains\n")
w("| | Laptop | Ibex (production) |")
w("|---|---|---|")
w(f"| CPU | {cpu} (arm64; {cores.group(1)} cores: {cores.group(2)} performance, {cores.group(3)} efficiency; {cores.group(4)} GB) | {env['machine.cpu'].replace(' 64-Core Processor', '')} (x86-64, Zen 2), as SN1 and E3's environment table give for the cluster's nodes; the fit records name the hosts (psmc-rs, E1 and NA12878: {rs_hosts}; C psmc, E1: {c_hosts}) but not their CPU model |")
w(f"| OS | macOS {macos} ({macbuild}), Darwin {darwin}, Apple libm | Linux x86-64 (E3's nodes: {env['machine.release'].replace('.0.1.x86_64', '')}, {env['machine.libc']}; not recorded for the E1/E2 jobs) |")
src_note = "clean" if dirty == 0 else f"{dirty} uncommitted files"
w(f"| psmc-rs sources | `src` {ids['src'][:7]}, `Cargo.toml` {ids['Cargo.toml'][:7]}, `Cargo.lock` {ids['Cargo.lock'][:7]}, {src_note} (HEAD {head[:7]} for both builds) | "
  + ("the same three ids" if ib_ids == ids else f"`src` {ib_ids['src'][:7]}, `Cargo.toml` {ib_ids['Cargo.toml'][:7]}, `Cargo.lock` {ib_ids['Cargo.lock'][:7]}")
  + f", {'clean' if rs_tool['source_dirty'] == '' else 'dirty'} (every E1/E2 fit-rs provenance record) |")
w(f"| rustc | {main_b['rustc']}, LLVM {main_b['llvm']}, {main_b['host']} — the default toolchain of `{main_b['cargo']}`; control build: {ctl_b['rustc']}, LLVM {ctl_b['llvm']} | {rs_tool['rustc'].replace(' x86_64', ', x86_64')} |")
w(f"| build flags | release profile, no RUSTFLAGS / CARGO_*RUSTFLAGS, `.cargo/config.toml`: {cargo_cfg}; `rustc --print cfg` has `neon` | none (`rustflags` empty, `target_features: {rs_tool['target_features']}`) |")
simd = Counter((r["simd_laptop"], r["simd_ibex"]) for r in rs)
assert len(simd) == 1
(sl, si), ns = simd.popitem()
w(f"| HMM kernels (`simd` in the fit JSON) | `{sl}` ({ns}/{n_rs} fits) | `{si}` ({ns}/{n_rs} fits) |")
w(f"| psmc-rs binary | `{main_b['version']}`, sha256 {main_b['sha'][:8]}… (rustc {main_b['rustc'].split()[0]}), {ctl_b['sha'][:8]}… (rustc {ctl_b['rustc'].split()[0]}); without the Mach-O UUID and the code signature, {main_b['digest'][:8]}… and {ctl_b['digest'][:8]}… | `{rs_tool['version']}`, sha256 {ib_bin[:8]}…"
  + (" (also the binary of E3's timings)" if env.get("tool.rs02.binary_sha256") == ib_bin else "") + " |")
w(f"| C psmc | 0.6.5-r74-dirty, sha256 {c_sha[:8]}… (the frozen `.production` run, fits of {', '.join(lap_c_days)}). Compiler and CFLAGS are in no provenance record. Its machine code (`__text`) is identical to the `psmc_O1` and `psmc_O` binaries beside it in `experiment/.tools/psmc_c` (so: -O1); the DWARF of the sibling ASan build names {producer}; it contains {n_fma} fused multiply-add instructions (in `psmc_update_hmm`, `hmm_forward`, `hmm_backward`, `hmm_expect`, `hmm_Q`, `hmm_Q0`, …), i.e. clang's default FP contraction was on | {c_tool['version']}, sha256 {c_tool['sha256'][:8]}…, {c_tool['cc'].replace('gcc (GCC) ', 'gcc ').replace(' 20231218', '')}, `{c_tool['cflags']}` (fits of {', '.join(c_days)}) |")
w("| threads | psmc-rs `--threads 1` (as recorded on Ibex); C psmc single-threaded | the same |")
w("")
w("The laptop binaries' whole-file sha256 depends on the target directory they were built in: rebuilt from the same sources by the same toolchain in another target directory, each differed in its Mach-O UUID (LC_UUID), and so in its ad-hoc code signature, and nowhere else. The sha256 with those bytes zeroed (`toolchain.txt`; `scripts/toolchain.py --digest <binary>` computes it) is the same for such a rebuild. Both still depend on the source paths a binary holds in its panic messages, under the builder's home directory (`~/.cargo`, `~/.rustup`), so another user's or machine's build is not expected to reproduce either.\n")
w(f"psmc-rs wall time per E1 fit (not a controlled timing): laptop {min(float(r['laptop_wall_seconds']) for r in rs1):.0f}–{mx(rs1, 'laptop_wall_seconds'):.0f} s (4 jobs at once, `nice 19`); Ibex {min(float(r['ibex_wall_seconds']) for r in rs1):.0f}–{mx(rs1, 'ibex_wall_seconds'):.0f} s (the production's fits shared their node). NA12878: laptop {float(na['laptop_wall_seconds']):.0f} s, Ibex {float(na['ibex_wall_seconds']):.0f} s.\n")

w("## 2. Inputs (step 1)\n")
w(f"- **E1 genomes: {I['n_psmcfa_eq_ibex_truth']}/{I['genomes']} byte-identical.** The SHA-256 of every laptop `genome.psmcfa` and `genome.vcf.gz` equals the one in the committed Ibex truth record ({I['n_psmcfa_eq_ibex_truth']}/40 and {I['n_vcf_gz_eq_ibex_truth']}/40) and in the Ibex sim provenance record (psmcfa {I['n_psmcfa_eq_ibex_sim_prov']}/40, vcf.gz {I['n_vcf_gz_eq_ibex_sim_prov']}/40, tabix index {I['n_tbi_eq_ibex_sim_prov']}/40).")
w(f"- **Per chromosome: {I['n_chromosomes_het_sha_match']}/{I['chromosomes']} match.** Recomputed from the laptop files, each chromosome's heterozygous-position checksum (SHA-256 of the 0-based positions as little-endian int64, read from the VCF) equals the Ibex truth record's `het_positions_sha256`, and so do its counts (bins and heterozygous bins from the psmcfa; heterozygous and derived homozygous sites and the last heterozygous position from the VCF): {I['n_chromosomes_all_counts_match']}/{I['chromosomes']}.")
w(f"- **Fit inputs.** The psmcfa checksum each Ibex fit recorded as its input equals the laptop file's ({I['n_psmcfa_eq_ibex_fit_rs_input']}/40 psmc-rs, {I['n_psmcfa_eq_ibex_fit_c_input']}/40 C psmc). The laptop C fits read the same files: their records' input checksums equal them ({I['n_psmcfa_eq_laptop_fit_c_input']}/40), their outputs still match their records ({I['n_laptop_c_output_eq_its_record']}/40) and the copies in `laptop_c/` ({S['c']['laptop_files_eq_laptop_c_copy']}/40), and their `.psmc` headers give the same n_seqs, sum_L and sum_n as Ibex's ({S['c']['headers_equal']}/40). The sim records' command, seeds and settings agree ({I['n_sim_record_key_equal']}/40), and so do the truth records' chromosome blocks ({I['n_truth_chromosomes_block_equal']}/40).")
w(f"- **NA12878** (`{I['NA12878']['path']}`, sha256 {I['NA12878']['sha256'][:12]}…) is the file both Ibex NA12878 fits read ({'yes' if I['NA12878']['eq_ibex_fit_rs_input'] and I['NA12878']['eq_ibex_fit_c_input'] else 'NO'}).")
w(f"- The production's Ibex fits are the files their provenance records name (psmc-rs {S['rs']['ibex_files_match_provenance']}/{n_rs}, C psmc {S['c']['ibex_files_match_record']}/40).\n")

w("## 3. Commands (step 3)\n")
n_ok = sum(rec["exit_status"] == 0 for rec in records)
w(f"Each laptop psmc-rs command is the Ibex provenance record's `key.command` with only the placeholders replaced: the binary (`<repo>/target/release/psmc-rs` → the laptop build), the input (`<runs>/E1/sims/...` → the identical laptop genome; NA12878 → this checkout's file) and the two outputs (→ `runs/...`). Every option is unchanged, including `--threads 1`, `--mstep-iters 1000`, `-N 25`, the pattern, `--t-max 15`, `--tr-ratio 5`, `--smooth-lambda 0`, `--bin-size 100`, `--mu`, `--gen-years` and `--no-progress` (`jobs.json` lists both commands; `records/` holds each run's command, checksums, exit status and times). {'All' if n_ok == len(records) else f'{n_ok} of'} {len(records)} exited 0.\n")
ex = next(j for j in JOBS if j["job"] == "E1/fit-rs/constant/rep01")
args = ex["command"][1:]
args = [a.replace("<repo>/experiment/.production/experiment/runs", "<laptop>").replace("<xplat>/runs/E1/fits/constant", "<out>") for a in args]
k = args.index("--smooth-lambda")
w("Example (E1 constant rep01):\n")
w("```")
w("psmc-rs " + " ".join(args[:k]) + " \\")
w("    " + " ".join(args[k:]))
w("```\n")

w("## 4. Results\n")
w("### 4.1 Byte identity\n")
w(f"- **psmc-rs, laptop vs Ibex: {n_bytes}/{n_rs} byte-identical** (JSON {S['rs']['json_bytes_identical']}/{n_rs}; JSON with its `simd` label set aside {S['rs']['json_identical_except_simd_label']}/{n_rs}; `.psmc` {S['rs']['psmc_bytes_identical']}/{n_rs}).")
w(f"- **C psmc, laptop build vs Ibex build: {S['c']['psmc_bytes_identical']}/40 byte-identical** (`.psmc`).")
w(f"- **Control, laptop rustc {ctl_b['rustc'].split()[0]} vs laptop rustc {main_b['rustc'].split()[0]}: {sum(yes(r['json_bytes_identical']) for r in ctl199)}/{len(ctl199)} JSON and {sum(yes(r['psmc_bytes_identical']) for r in ctl199)}/{len(ctl199)} `.psmc` byte-identical.**")
w(f"- **Control, the same laptop binary run twice** ({', '.join(name(r) for r in rep)}): {rep_same}/{len(rep)} byte-identical (JSON and `.psmc`).")
w("")
w("### 4.2 Where the runs part\n")
odd = [r for r in rs if float(r["loglike_iter0_rel"]) != 0]
ulps = {1: "one unit", 2: "two units", 3: "three units"}
odd_txt = "; ".join(f"{'E1 ' if r['exp'] == 'E1' else ''}{name(r)}, it differs by {ulps.get(int(r['loglike_iter0_ulps']), r['loglike_iter0_ulps'] + ' units')} in the last place ({f(r['loglike_iter0_rel'])} relative)" for r in odd)
fd = Counter((int(r["psmc_first_diff_round"]), r["psmc_first_diff_tag"]) for r in rs)
fd_rounds = sorted({k[0] for k in fd})
fl = S["rs"]["loglike_first_diff_iter"]
w(f"- **psmc-rs.** The log-likelihood after 0 iterations (the E-step at the starting parameters) is bit-identical in {S['rs']['loglike_iter0_identical']}/{n_rs} fits"
  + (f"; in the other{'s' if len(odd) > 1 else ''}, {odd_txt}" if odd else "") + ". "
  + "The first log-likelihood that differs is that after iteration " + ", ".join(f"{k} ({plural(v, 'fit')})" for k, v in sorted(fl.items(), key=lambda kv: int(kv[0])))
  + f", at the level of the last digits; the first differing `.psmc` line is mostly the M-step's evaluation count (`IT`) of iterations {fd_rounds[0]} to {fd_rounds[-1]} ("
  + ", ".join(f"round {k[0]} {k[1]}: {v}" for k, v in sorted(fd.items(), key=lambda kv: (-kv[1], kv[0])))
  + "). Rust does not contract floating-point operations, and per SN1 section 8 (and `src/hmm/simd.rs`) the NEON and the AVX2+FMA kernels fuse the same multiply-adds and no sum changes its order with the vector width, which the identical iteration-0 log-likelihoods bear out; what remains to differ is the system math library (Apple libm vs glibc, whose `exp`/`log` can round differently in the last bit) and any other architecture-specific code path. This run does not separate them. The M-step, stopping on a flat optimum, turns the last-digit differences into the parameter differences below.")
cfd = S["c"]["psmc_first_diff"]
w(f"- **C psmc.** In {'all 40' if sum(cfd.values()) == 40 and len(cfd) == 1 else 'the 40'} fits the first differing line is the `IT` count of iteration 1's Hooke–Jeeves M-step ({', '.join(f'{k}: {v}' for k, v in cfd.items())}); iteration 1's LK and QD-before lines are identical in {S['c']['round1_lk_and_qd_before_identical']}/40 (the E-step at the starting parameters agrees to C's printed precision, 6 decimals), and from there the evaluation counts differ in a median {int(med(c, 'mstep_counts_differ'))} of 25 iterations. The two C builds differ in compiler, optimisation level, FP contraction (the laptop binary fuses multiply-adds in the E-step and the M-step objective; Ibex's does not) and math library, which this comparison cannot separate.\n")

w("### 4.3 Agreement (min / median / max over the 40 E1 fits; NA12878 single fit)\n")
w("Relative differences are |a − b| / |b| with Ibex as b. Δlog10 Ne is compared interval by interval (`metrics.interval_diff`; Ne_k = λ_k·N0, N0 ∝ θ): 'all intervals' is the pre-registered d (all 64 intervals, the out-of-window first λ included); 'window' uses the committed informative windows of `E1_rs_vs_c.csv` (NA12878: `E2_rs_vs_c.csv`), weighted by log time as in E1. C psmc's parameters come from its `PA` lines (9 decimals) and its last log-likelihood is iteration 24's (C prints none for the final parameters); psmc-rs's from its JSON (full precision).\n")
labels = [
    ("theta_rel", "θ"), ("rho_rel", "ρ"), ("t_max_rel", "t_max"), ("lambda_max_rel", "largest free λ"),
    ("lambda_median_rel", "median free λ"), ("param_max_rel", "largest of all (pre-registered p)"),
    ("dlog10ne_max_all_intervals", "max \\|Δlog10 Ne\\|, all intervals (pre-registered d)"),
    ("dlog10ne_max_window", "max \\|Δlog10 Ne\\|, window"), ("dlog10ne_rmse_window", "RMSE Δlog10 Ne, window"),
    ("dlog10_t_boundaries_max", "max \\|Δlog10 t_k\\| (time scale)"),
    ("loglike_rel_final", "final log-likelihood (pre-registered)"),
    ("loglike_rel_max_trace", "log-likelihood, worst iteration"), ("loglike_iter0_rel", "log-likelihood, iteration 0"),
]
show199 = bool(r199_1) and ctl_same != len(ctl199)
hdr = "| quantity | psmc-rs laptop vs Ibex | C psmc laptop vs Ibex | psmc-rs vs C psmc (both Ibex) | NA12878 psmc-rs laptop vs Ibex |"
if show199:
    hdr += " psmc-rs laptop (rustc 1.99.0) vs Ibex |"
w(hdr)
w("|" + "---|" * (hdr.count("|") - 1))
for k, label in labels:
    line = f"| {label} | {rng(S['rs']['E1'].get(k))} | {rng(S['c']['E1'].get(k))} | {rng(S['rs_vs_c_ibex']['E1'].get(k))} | {f(na[k])} |"
    if show199:
        line += f" {rng(S['rs_rustc1.99']['E1'].get(k))} |"
    w(line)
w("")
top_rs = max(rs1, key=lambda r: float(r["dlog10ne_rmse_window"]))
w(f"- C psmc's largest parameter difference ({float(c_top['param_max_rel']):.2f}) and d ({float(c_top['dlog10ne_max_all_intervals']):.2f}) are {name(c_top)}'s first λ, which runs away out of the window in both builds ({RA['c_lambda_laptop']:.3g} on the laptop, {RA['c_lambda_ibex']:.3g} on Ibex: {f(c_top['param_max_rel'])} relative, {f(c_top['dlog10ne_max_all_intervals'])} in log10; inside the window that fit agrees to {f(c_top['dlog10ne_max_window'])}). Without that fit: largest parameter difference {f(mx(c_rest, 'param_max_rel'))}, d {f(mx(c_rest, 'dlog10ne_max_all_intervals'))}. "
  + f"(psmc-rs bounds every free λ at 1e4, so the same λ stops at the bound on both platforms, at {RA['rs_lambda_laptop']!r} on the laptop and {RA['rs_lambda_ibex']!r} on Ibex; on the way there, at iteration {RA['rs_rel_max_round']}, it differed by {f(RA['rs_rel_max_over_rounds'])} relative between them.)"
  + (f" The same replicate holds psmc-rs's largest differences: its iteration-0 log-likelihood is the one that differs, and its fit differs by {f(top_rs['param_max_rel'])} on {lam(top_rs['param_max_rel_name'])}, by a d of {f(top_rs['dlog10ne_max_all_intervals'])} on λ{top_rs['dlog10ne_argmax_lambda']}, {'inside' if yes(top_rs['dlog10ne_argmax_in_window']) else 'outside'} the window, and by an RMSE of {f(top_rs['dlog10ne_rmse_window'])} over the window, the largest of the 40."
     if name(top_rs) == name(c_top) and odd and name(odd[0]) == name(top_rs) else ""))
for comp, label in (("psmc-rs laptop vs Ibex", "psmc-rs"), ("C psmc laptop vs Ibex", "C psmc")):
    rr = [r for r in prm if r["comparison"] == comp and r["exp"] == "E1"]
    lm = {i: max(float(r[f"lam{i}"]) for r in rr) for i in range(28)}
    top = sorted(lm, key=lm.get, reverse=True)
    rest = [i for i in range(28) if i not in top[:3]]
    w(f"- Each λ, {label} (largest relative difference over the 40 fits; `params_rel_per_fit.csv`): "
      + ", ".join(f"λ{i} {f(lm[i])}" for i in top[:3]) + f"; the other 25 at most {f(max(lm[i] for i in rest))}.")
bx = Counter(r["param_max_rel_name"] for r in rs1 if r["model"] in ("bottleneck", "expansion"))
cz = Counter(r["param_max_rel_name"] for r in rs1 if r["model"] in ("constant", "zigzag"))
top_d = max(rs1, key=lambda r: float(r["dlog10ne_max_all_intervals"]))
w(f"- psmc-rs's largest relative parameter difference is, in the 20 bottleneck and expansion fits: {counts(bx)}; in the 20 constant and zigzag fits: {counts(cz)}. "
  f"Its largest d, {f(top_d['dlog10ne_max_all_intervals'])}, is {name(top_d)}'s λ{top_d['dlog10ne_argmax_lambda']}, "
  + ("inside the window." if yes(top_d["dlog10ne_argmax_in_window"]) else f"outside the window (inside: {f(top_d['dlog10ne_max_window'])}).") + "\n")

w("### 4.4 Per model (median [max] over 10 replicates)\n")
w("| model | RMSE window: rs xplat | C xplat | rs vs C (Ibex) | max window: rs xplat | C xplat | rs vs C | d all: rs xplat | C xplat | rs vs C |")
w("|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    a, b, r = S["rs"]["E1_per_model"][m], S["c"]["E1_per_model"][m], S["rs_vs_c_ibex"]["E1_per_model"][m]

    def mm(st):
        return f"{f(st['median'])} [{f(st['max'])}]"
    w(f"| {m} | {mm(a['dlog10ne_rmse_window'])} | {mm(b['dlog10ne_rmse_window'])} | {mm(r['dlog10ne_rmse_window'])} | "
      f"{mm(a['dlog10ne_max_window'])} | {mm(b['dlog10ne_max_window'])} | {mm(r['dlog10ne_max_window'])} | "
      f"{mm(a['dlog10ne_max_all_intervals'])} | {mm(b['dlog10ne_max_all_intervals'])} | {mm(r['dlog10ne_max_all_intervals'])} |")
w("")
w("### 4.5 Side by side (per E1 replicate; `side_by_side_E1.csv`)\n")
sc = S["side_by_side_counts"]
w(f"- psmc-rs's cross-platform window RMSE is below its RMSE against C psmc in **{sc['rs_xplat_rmse_below_rs_vs_c']}/{sc['n']}** replicates; ratio (rs vs C) / (rs cross-platform), min / median / max: {ratio('ratio_rmse_rsvsc_over_rs_xplat')}.")
w(f"- psmc-rs's cross-platform RMSE is below C psmc's cross-build RMSE in **{sc['rs_xplat_rmse_below_c_xplat']}/{sc['n']}**; ratio (C xplat) / (rs xplat): {ratio('ratio_rmse_c_xplat_over_rs_xplat')}.")
w(f"- C psmc's cross-build RMSE is below the psmc-rs-vs-C RMSE in {sc['c_xplat_rmse_below_rs_vs_c']}/{sc['n']}; ratio (rs vs C) / (C xplat): {ratio('ratio_rmse_rsvsc_over_c_xplat')} (the noise floor of B7: psmc-rs vs C is a few times C vs C).")
w(f"- Log-likelihood, worst iteration (median): psmc-rs cross-platform {f(S['side_by_side']['loglike_rel_max_rs_laptop_vs_ibex']['median'])}, C cross-build {f(S['side_by_side']['loglike_rel_max_c_laptop_vs_ibex']['median'])}, psmc-rs vs C {f(S['side_by_side']['loglike_rel_max_rs_vs_c_ibex']['median'])}.\n")

w("### 4.6 The tables, written again on the laptop (`reports.py`)\n")
code_same = R["code_identical"]
w(f"E1, E2, E4 and E5's reports (`python -m experiment.pipeline report`) ran on the laptop on a copy of the committed runs directory, its tables removed, and wrote the {R['tables']} tables again; the committed ones are those the production's reports wrote on Ibex ({', '.join(sorted({f'{v['ibex_host']}, SLURM job {v['ibex_slurm_job_id']}' for v in R['code'].values()}))})"
  + (", and each report's identity on the laptop (the digest of its code, Python's and the packages' versions) is that of its Ibex record (4/4)" if code_same else "; the reports' identities on the laptop differ from Ibex's") + ". `reports_laptop_vs_ibex.csv` compares them cell by cell:\n")
w(f"- {R['tables_bytes_identical']}/{R['tables']} tables byte-identical, all {R['tables_same_shape']} with the same rows and columns; {R['cells_differing']} of {R['cells']} cells differ, none of them text ({R['nonnumeric_differing']}).")
w(f"- Every value of magnitude 1e-3 or more agrees to a relative {f(R['max_rel_ge_1e-3'])} at most; smaller values to {f(R['max_abs_lt_1e-3'])} absolute.")
w(f"- The differences between psmc-rs and C psmc ({R['rs_vs_c_cells']} values of the `*rs_vs_c*` columns of {', '.join('`' + t.split('/')[1] + '`' for t in R['rs_vs_c_tables'])}, of order 1e-5) agree to a relative {f(R['rs_vs_c_max_rel'])} ({f(R['rs_vs_c_max_abs'])} absolute).")
w(f"- The same code on the laptop recomputes the committed `E1_rs_vs_c.csv` (`compare.py`'s `rs_vs_c_ibex.csv`) to {f(S['repro_committed_rs_vs_c_max_abs_error'])} absolute.\n")

w("## 5. Pre-registered tiers\n")
w("Tiers as defined in the review's `xplatform.py`, unchanged: **A** output bytes identical; **B** p ≤ 1e-6, d ≤ 1e-6 and final log-likelihood ≤ 1e-8 (relative); **C** d ≤ 1e-3 and 50·d ≤ the reference psmc-rs-vs-C RMSE; otherwise **fail**. Bytes for A: psmc-rs's JSON (as for the bootstrap replicates of the pre-registered script), C psmc's `.psmc`.\n")
ref = S["references"]
w("The pre-registration leaves the tier-C reference unspecified (review B2). Primary: the per-model mean psmc-rs-vs-C RMSE of the committed `E1_table1.csv` (`rmse_rs_vs_c_per_interval_mean`, interval by interval after the B1 fix: "
  + ", ".join(f"{m} {f(ref['model_mean_primary'][m])}" for m in MODELS)
  + f"); for NA12878 the mean over all 40 E1 replicates ({f(ref['all40_mean'])}), as the pre-registered script did for E2. These are smaller (stricter) than the pre-B1 references the review used.\n")
w("| tier-C reference | psmc-rs E1 (40) | psmc-rs NA12878 | C psmc E1 (40) |" + (" psmc-rs rustc 1.99.0 E1 (40) |" if show199 else ""))
w("|---|---|---|---|" + ("---|" if show199 else ""))
for k, label in (("tier", "**primary**: per-model mean (E2: all-40 mean)"),
                 ("tier_vs_pair_rmse", f"the same replicate's rs-vs-C RMSE (NA12878: its own, {f(ref['NA12878_pair'])})"),
                 ("tier_vs_all40_mean", "mean of all 40 E1 replicates"),
                 ("tier_vs_min_model_mean", "smallest per-model mean (constant)"),
                 ("tier_vs_preB1_ref", "the review's superseded pre-B1 references (sampled curves; larger)"),
                 ("tier_window_d_DEVIATION", "*deviation*: d over the window's intervals only"),
                 ("tier_bytes_excl_simd_label_DEVIATION", "*deviation*: tier A ignores the JSON's `simd` label")):
    line = f"| {label} | {tiers(rs1, k)} | {tiers([na], k)} | {tiers(c, k) if k in c[0] else '–'} |"
    if show199:
        line += f" {tiers(r199_1, k)} |"
    w(line)
w("")
notb = [r for r in rs1 if r["tier"] != "B"]
p_ok = sum(float(r["param_max_rel"]) <= 1e-6 for r in rs1)
d_ok = sum(float(r["dlog10ne_max_all_intervals"]) <= 1e-6 for r in rs1)
l_ok = sum(float(r["loglike_rel_final"]) <= 1e-8 for r in rs1)
on_p = [r for r in notb if float(r["param_max_rel"]) > 1e-6]
w(f"- **psmc-rs (primary): B {nb['B']}, C {nb['C']}, fail {nb['fail']} of 40; NA12878: {na['tier']}.** "
  + (f"Every one of the {len(notb)} fits that miss B exceeds p ≤ 1e-6" if len(on_p) == len(notb) else f"{len(on_p)} of the {len(notb)} fits that miss B exceed p ≤ 1e-6")
  + f" ({sum(float(r['dlog10ne_max_all_intervals']) <= 1e-6 for r in on_p)} on p alone, {sum(float(r['dlog10ne_max_all_intervals']) > 1e-6 for r in on_p)} also on d; the parameter: {counts(Counter(r['param_max_rel_name'] for r in on_p))}); the log-likelihood test passes in {l_ok}/40 (p ≤ 1e-6 in {p_ok}/40, d ≤ 1e-6 in {d_ok}/40). By model, B: "
  + ", ".join(f"{m} {sum(r['tier'] == 'B' for r in rs1 if r['model'] == m)}/10" for m in MODELS) + "."
  + (f" NA12878 misses B on p ({f(na['param_max_rel'])} on {lam(na['param_max_rel_name'])}; d {f(na['dlog10ne_max_all_intervals'])}, reference/d {float(na['ratio_ref_over_d']):.0f})." if na["tier"] != "B" else ""))


def fail_txt(r):
    t = f"{name(r)} (p {f(r['param_max_rel'])} on {lam(r['param_max_rel_name'])}, d {f(r['dlog10ne_max_all_intervals'])} on λ{r['dlog10ne_argmax_lambda']}, ratio {float(r['ratio_ref_over_d']):.0f}"
    if r["tier_window_d_DEVIATION"] != r["tier"]:
        t += f"; outside the window, and over the window it would be {r['tier_window_d_DEVIATION']}"
    return t + ")"


w(f"- The {plural(len(fails), 'fail')} (reference / d needs ≥ 50): " + "; ".join(fail_txt(r) for r in fails) + ".")
cb = Counter(r["tier"] for r in c)
pre = Counter(r["tier_vs_preB1_ref"] for r in c)
others = [k for k in ("tier", "tier_vs_pair_rmse", "tier_vs_all40_mean", "tier_vs_min_model_mean", "tier_window_d_DEVIATION")]
c_all_fail = all(Counter(r[k] for r in c) == Counter({"fail": 40}) for k in others)
w(f"- **C psmc: {tiers(c)} of 40** " +("under every reference considered" if c_all_fail else "under the primary reference")
  + (f" except the review's superseded pre-B1 one ({tiers(c, 'tier_vs_preB1_ref')})" if pre != cb else "")
  + f". It fails B on both p (≥ {f(S['c']['E1']['param_max_rel']['min'])}) and d (≥ {f(S['c']['E1']['dlog10ne_max_all_intervals']['min'])}), and C because 50·d exceeds the reference (reference/d at most {f(mx(c, 'ratio_ref_over_d'))}). Its log-likelihoods would pass B ({f(S['c']['E1']['loglike_rel_final']['max'])} at most). Its fits are those of the review's B7 for these 40 genomes (there: A 0, B 0, C 1, fail 43 of 44 with the pre-B1 reference).")
cut = {m: ref["model_mean_primary"][m] / 50 for m in MODELS}
nonnested = [r for r in rs1 if r["tier"] == "B" and float(r["dlog10ne_max_all_intervals"]) * 50 > float(r["ref_rmse_primary"])]
w("- Note on the tiers' structure: with the corrected references, tier C's d-test (d ≤ reference/50: "
  + ", ".join(f"{f(cut[m])} {m}" for m in MODELS) + ") is stricter than tier B's d ≤ 1e-6 for the "
  + " and ".join(m for m in MODELS if cut[m] < 1e-6) + " model" + ("s" if sum(cut[m] < 1e-6 for m in MODELS) > 1 else "")
  + ", so the tiers are not nested"
  + (f": {', '.join(name(r) for r in nonnested)} {'are' if len(nonnested) > 1 else 'is'} B although {'their' if len(nonnested) > 1 else 'its'} d would fail the C test.\n" if nonnested else "; no fit here is B with a d that would fail the C test.\n"))

w(f"## 6. Control: the laptop build with Ibex's rustc ({ctl_b['rustc'].split()[0]})\n")
w(f"The same sources built on the laptop by rustc {ctl_b['rustc'].split()[0]} ({ctl_b['rustc'].split(' (')[1].rstrip(')')}, the commit of Ibex's rustc, {rs_tool['rustc'].split()[1].strip('()')}; LLVM {ctl_b['llvm']}) gave {ctl_same}/{len(ctl199)} fits byte-identical to the rustc {main_b['rustc'].split()[0]} build's (JSON and `.psmc`). "
  + ("The rustc version therefore does not change psmc-rs's output on the laptop, and the laptop-vs-Ibex differences above are those of the platform (CPU architecture and kernels, operating system and its math library), not of the compiler version." if ctl_same == len(ctl199) else
     "The rustc version changes the output for some fits; the comparison with Ibex at the same rustc is in `rs_laptop_rustc1.99_vs_ibex.csv` and in the last columns of the tables above."))
w("")

w("## 7. Deviations from the pre-registration\n")
w(f"1. **Tier A for psmc-rs** is tested on the JSON, as pre-registered, but the JSON records the kernels that ran (`simd`: `{sl}` vs `{si}`), so it can never be byte-identical across these two platforms. Also testing the JSON without that label, and the `.psmc` file, changes nothing: "
  + ("no fit is identical either way." if S["rs"]["json_identical_except_simd_label"] == 0 and S["rs"]["psmc_bytes_identical"] == 0 else f"{S['rs']['json_identical_except_simd_label']} and {S['rs']['psmc_bytes_identical']} fits are."))
w("2. **Tier-C reference.** Unspecified in the pre-registration; the primary choice (per-model mean, matched intervals) is the corrected version (B1) of what the review's script used. All alternatives are reported (table in section 5).")
moved = [r for r in rs1 + [na] if r["tier_window_d_DEVIATION"] != r["tier"]]
moved_c = [r for r in c if r["tier_window_d_DEVIATION"] != r["tier"]]
w("3. **d over the window only** (excluding an out-of-window first λ) is reported as a labelled deviation, never as the result; it "
  + ("moves psmc-rs " + ", ".join(f"{name(r)} from {r['tier']} to {r['tier_window_d_DEVIATION']}" for r in moved) if moved else "changes no psmc-rs fit's tier")
  + (" and changes nothing for C psmc." if not moved_c else f" and moves {len(moved_c)} C psmc fits."))
w(f"4. **Compiler version.** The laptop build used the default toolchain (rustc {main_b['rustc'].split()[0]}), not Ibex's {ctl_b['rustc'].split()[0]}; the rustc {ctl_b['rustc'].split()[0]} control (section 6) addresses it.")
w("5. **NA12878 window and reference.** Its window is E2's committed one; its tier-C reference is the all-40 E1 mean (the pre-registered rule for E2 fits).\n")

w("## 8. Caveats\n")
w("- **Threads:** identical on both platforms (psmc-rs `--threads 1` as recorded; C psmc is single-threaded). The determinism control (one binary, two runs, byte-identical) shows the laptop differences are not run-to-run noise.")
w(f"- **Laptop C psmc** is the frozen `.production` run's binary and fits; its compiler and flags are not in its provenance. The evidence above (machine code identical to the -O1 builds; {producer.split(' (')[0]} from a sibling build's DWARF; FP contraction on) is circumstantial. Not done: refitting the 40 genomes with a laptop C build with `-ffp-contract=off` (the laptop's C fits compared here took {min(float(r['laptop_wall_seconds']) for r in c) / 60:.0f}–{mx(c, 'laptop_wall_seconds') / 60:.0f} min each), which would separate FMA contraction from the compiler and libm.")
w("- **Ibex CPU and OS** for the E1/E2 jobs are not in their records; the fits' `simd: avx2+fma` confirms an AVX2+FMA CPU.")
w(f"- **psmc-rs source:** the same on both platforms (`src` {ids['src'][:7]}, the production's), which removes the review's m21 confound (the review's laptop psmc-rs was df00bf2e). The laptop's C fits and simulated genomes come from an earlier laptop run, `.production` (they do not depend on psmc-rs).")
w("- **Scope:** E1's 40 fits of each tool and NA12878's psmc-rs fit, and the tables of E1, E2, E4 and E5. Not refitted on the laptop: NA12878's C fit, the turtle genomes, bootstraps, E4–E7.\n")

w("## 9. Files (in this directory)\n")
for fname, desc in (
    ("`inputs_E1_genomes.csv`, `inputs_E1_chromosomes.csv`, `inputs.json`", f"step 1 ({I['genomes']} genomes, {I['chromosomes']} chromosomes, NA12878)"),
    ("`rs_laptop_vs_ibex.csv`", f"psmc-rs laptop (rustc {main_b['rustc'].split()[0]}) vs Ibex, {len(rs)} rows, with tiers"),
    ("`c_laptop_vs_ibex.csv`", f"C psmc laptop vs Ibex, {len(c)} rows, with tiers"),
    ("`rs_vs_c_ibex.csv`", f"psmc-rs vs C psmc on Ibex, recomputed with the same code (matches the committed `E1_rs_vs_c.csv` to {f(S['repro_committed_rs_vs_c_max_abs_error'])})"),
    ("`side_by_side_E1.csv`", "the three agreements per E1 replicate, and their ratios"),
    ("`params_rel_per_fit.csv`", "relative difference of θ, ρ, t_max and each of the 28 free λ, per fit"),
    ("`rs_laptop_rustc1.99_vs_ibex.csv`, `control_rustc1.99_vs_rustc1.97_laptop.csv`, `control_repeat_laptop.csv`", "the controls"),
    ("`runs-rustc1.99/`, `runs-repeat/`", f"the controls' fits ({len(ctl199)} by the rustc {ctl_b['rustc'].split()[0]} build{', byte-identical to `runs/`' if ctl_same == len(ctl199) else ''}; {len(rep)} by the same binary run again); "
     "`jobs_rustc1.99.json`, `jobs_repeat.json`: their job lists; `records-rustc1.99/`, `records-repeat/`: their records"),
    ("`reports_laptop_vs_ibex.csv`, `reports.json`", "the tables written again on the laptop, table by table (section 4.6)"),
    ("`summary.json`", "every number above; `toolchain.txt`: the laptop, its two rustc builds, the binaries' checksums and the C psmc binary's forensics"),
    ("`jobs.json`", "the Ibex and the laptop command of each job"),
    ("`runs/`", "the laptop psmc-rs fits (JSON and `.psmc`); `records/`: one record per laptop run (command, checksums, exit status, times)"),
    ("`laptop_c/`", "the laptop C psmc fits compared here, copied from the frozen laptop run `experiment/.production` (not in the repository)"),
    ("not kept", "logs, the HTML reports, the binaries (`bin/`) and the build directories"),
):
    w(f"- {fname}: {desc}")
(X / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
