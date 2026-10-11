"""Write xplat/summary.md from summary.json, inputs.json, toolchain.txt and the CSVs."""
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

X = Path(__file__).resolve().parent.parent
S = json.loads((X / "summary.json").read_text())
I = json.loads((X / "inputs.json").read_text())
MODELS = ("constant", "bottleneck", "expansion", "zigzag")


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
tool = (X / "toolchain.txt").read_text()

c_noexp5 = [r for r in c if not (r["model"] == "expansion" and r["replicate"] == "5")]
c_e5 = next(r for r in c if r["model"] == "expansion" and r["replicate"] == "5")
rsc_noexp5 = [r for r in rsc if not (r["model"] == "expansion" and r["replicate"] == "5")]
fails = [r for r in rs1 if r["tier"] == "fail"]


def mx(rr, k):
    return max(float(r[k]) for r in rr)


def med(rr, k):
    return statistics.median(float(r[k]) for r in rr)


def ratio(k):
    v = [float(s[k]) for s in side]
    return f"{f(min(v))} / {f(statistics.median(v))} / {f(max(v))}"


out = []
w = out.append
w("# Cross-platform comparison: laptop vs Ibex (data for Supplementary Note 1, section 9)\n")
w("Everything here was computed in this session under `xplat/`; nothing in the repository was changed. "
  "Scripts: `scripts/make_jobs.py`, `run_job.py`, `inputs.py`, `compare.py`, `tables.py`, `make_summary.py`.\n")

w("## Headline\n")
e1 = S["rs"]["E1"]
ce = S["c"]["E1"]
w(f"- **Inputs identical:** the laptop's 40 E1 genomes are byte-identical to Ibex's (psmcfa, VCF.gz and index: 40/40 each; 400/400 per-chromosome checksums and counts); NA12878's input too.")
w(f"- **psmc-rs, same sources, laptop (arm64 NEON, macOS) vs Ibex (x86-64 AVX2+FMA, Linux): 0/41 outputs byte-identical.** Median (max) over the 40 E1 fits: θ {f(e1['theta_rel']['median'])} ({f(e1['theta_rel']['max'])}) relative, any parameter {f(e1['param_max_rel']['median'])} ({f(e1['param_max_rel']['max'])}), log10 Ne per interval {f(e1['dlog10ne_max_all_intervals']['median'])} ({f(e1['dlog10ne_max_all_intervals']['max'])}), window RMSE {f(e1['dlog10ne_rmse_window']['median'])} ({f(e1['dlog10ne_rmse_window']['max'])}), final log-likelihood {f(e1['loglike_rel_final']['median'])} ({f(e1['loglike_rel_final']['max'])}). NA12878: parameters {f(na['param_max_rel'])}, log10 Ne {f(na['dlog10ne_max_all_intervals'])}, log-likelihood {f(na['loglike_rel_final'])}.")
w(f"- **Pre-registered tiers, psmc-rs:** B {Counter(r['tier'] for r in rs1)['B']}, C {Counter(r['tier'] for r in rs1)['C']}, **fail {len(fails)}** of 40 (NA12878: {na['tier']}). **C psmc (laptop Apple clang build vs Ibex gcc -O3 build): fail 40/40.**")
def rq(k, q):
    v = [float(x[k]) for x in side]
    x = min(v) if q == "min" else statistics.median(v)
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


w(f"- **Side by side (window RMSE, per replicate):** psmc-rs's cross-platform difference is {rq('ratio_rmse_rsvsc_over_rs_xplat', 'med')} times (median; at least {rq('ratio_rmse_rsvsc_over_rs_xplat', 'min')}) smaller than its difference from C psmc, and {rq('ratio_rmse_c_xplat_over_rs_xplat', 'med')} times (median; at least {rq('ratio_rmse_c_xplat_over_rs_xplat', 'min')}) smaller than C psmc's own cross-build difference, which is in turn {rq('ratio_rmse_rsvsc_over_c_xplat', 'med')} times smaller than psmc-rs vs C (median; C's cross-build RMSE is the larger in {40 - S['side_by_side_counts']['c_xplat_rmse_below_rs_vs_c']}/40).")
w(f"- **Controls:** the same sources built on the laptop by Ibex's rustc (1.99.0) give byte-identical output to the rustc 1.97.1 build (41/41), and one binary run twice gives identical bytes (2/2): the differences are the platform's (architecture, kernels, math library), not the compiler version's or run-to-run noise.\n")
w("## 1. Machines and toolchains\n")
w("| | Laptop | Ibex (production) |")
w("|---|---|---|")
w("| CPU | Apple M1 Pro (arm64; 8 cores: 6 performance, 2 efficiency; 16 GB) | AMD EPYC 7702 (x86-64, Zen 2), as SN1 and E3's environment table give for the cluster's nodes; the E1/E2 fit records name the hosts (cn514-11-r for all 80 E1 fits, SLURM job 53386770; cn514-14-r for NA12878, job 53452276) but not their CPU model |")
w("| OS | macOS 27.0 (26A428), Darwin 27.0.0, Apple libm | Linux x86-64 (E3's nodes: 5.14.0-427.20.1.el9_4, glibc 2.34; not recorded for the E1/E2 jobs) |")
w("| psmc-rs sources | `src` 4c0d777, `Cargo.toml` 7ff6ee3, `Cargo.lock` b552a2a, clean (HEAD 980bab8 for the rustc 1.97.1 build; 2b9a3d8, a concurrent commit that touches only E2's Papuan outputs, for the rustc 1.99.0 control build) | the same three ids, clean (every E1/E2 fit-rs provenance record) |")
w("| rustc | 1.97.1 (8bab26f4f 2026-07-14), LLVM 22.1.6, aarch64-apple-darwin — the default toolchain of `cargo build --release`; control build: 1.99.0 (b940084d7eb6), LLVM 23.1.1 | 1.99.0 (b940084d7eb6), x86_64-unknown-linux-gnu |")
w("| build flags | release profile, no RUSTFLAGS / CARGO_*RUSTFLAGS, `.cargo/config.toml` comments only; `rustc --print cfg` has `neon` | none (`rustflags` empty, `target_features: none`) |")
w("| HMM kernels (`simd` in the fit JSON) | `neon` (41/41 fits) | `avx2+fma` (41/41 fits) |")
w("| psmc-rs binary | `psmc-rs 0.2.0`, sha256 b8c48afa… (rustc 1.97.1), d4d6d08c… (rustc 1.99.0) | `psmc-rs 0.2.0`, sha256 2085fe3a… |")
w("| C psmc | 0.6.5-r74-dirty, sha256 c40db274… (the frozen `.production` run, fits of 2026-10-06). Compiler and CFLAGS are in no provenance record. Its machine code (`__text`) is identical to the `psmc_O1` and `psmc_O` binaries beside it in `experiment/.tools/psmc_c` (so: -O1); the DWARF of the sibling ASan build names Apple clang 21.0.0 (clang-2100.3.34.2); it contains 23 fused multiply-add instructions (in `psmc_update_hmm`, `hmm_forward`, `hmm_backward`, `hmm_expect`, `hmm_Q`, `hmm_Q0`, …), i.e. clang's default FP contraction was on | 0.6.5-r74-dirty, sha256 d5e0e160…, gcc 11.4.1 (Red Hat 11.4.1-3), `-g -Wall -O3 -march=x86-64-v3 -ffp-contract=off -Wno-unused-function` (fits of 2026-10-07) |")
w("| threads | psmc-rs `--threads 1` (as recorded on Ibex); C psmc single-threaded | the same |")
w("")
w(f"psmc-rs wall time per E1 fit (not a controlled timing): laptop {min(float(r['laptop_wall_seconds']) for r in rs1):.0f}–{mx(rs1, 'laptop_wall_seconds'):.0f} s (4 jobs at once, `nice 19`); Ibex {min(float(r['ibex_wall_seconds']) for r in rs1):.0f}–{mx(rs1, 'ibex_wall_seconds'):.0f} s. NA12878: laptop {float(na['laptop_wall_seconds']):.0f} s, Ibex {float(na['ibex_wall_seconds']):.0f} s.\n")

w("## 2. Inputs (step 1)\n")
w(f"- **E1 genomes: {I['n_psmcfa_eq_ibex_truth']}/{I['genomes']} byte-identical.** The SHA-256 of every laptop `genome.psmcfa` and `genome.vcf.gz` equals the one in the committed Ibex truth record ({I['n_psmcfa_eq_ibex_truth']}/40 and {I['n_vcf_gz_eq_ibex_truth']}/40) and in the Ibex sim provenance record (psmcfa {I['n_psmcfa_eq_ibex_sim_prov']}/40, vcf.gz {I['n_vcf_gz_eq_ibex_sim_prov']}/40, tabix index {I['n_tbi_eq_ibex_sim_prov']}/40).")
w(f"- **Per chromosome: {I['n_chromosomes_het_sha_match']}/{I['chromosomes']} match.** Recomputed from the laptop files, each chromosome's heterozygous-position checksum (SHA-256 of the 0-based positions as little-endian int64, read from the VCF) equals the Ibex truth record's `het_positions_sha256`, and so do its counts (bins and heterozygous bins from the psmcfa; heterozygous and derived homozygous sites and the last heterozygous position from the VCF): {I['n_chromosomes_all_counts_match']}/400.")
w(f"- **Fit inputs.** The psmcfa checksum each Ibex fit recorded as its input equals the laptop file's ({I['n_psmcfa_eq_ibex_fit_rs_input']}/40 psmc-rs, {I['n_psmcfa_eq_ibex_fit_c_input']}/40 C psmc). The laptop C fits read the same files: their records' input checksums equal them ({I['n_psmcfa_eq_laptop_fit_c_input']}/40), their outputs still match their records ({I['n_laptop_c_output_eq_its_record']}/40), and their `.psmc` headers give the same n_seqs, sum_L and sum_n as Ibex's ({S['c']['headers_equal']}/40). The sim records' command, seeds and settings agree ({I['n_sim_record_key_equal']}/40), and so do the truth records' chromosome blocks ({I['n_truth_chromosomes_block_equal']}/40).")
w(f"- **NA12878** (`experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz`, sha256 {I['NA12878']['sha256'][:12]}…) is the file both Ibex NA12878 fits read ({'yes' if I['NA12878']['eq_ibex_fit_rs_input'] and I['NA12878']['eq_ibex_fit_c_input'] else 'NO'}).")
w(f"- The committed Ibex fits are the files their provenance records name (psmc-rs {S['rs']['ibex_files_match_provenance']}/41, C psmc {S['c']['ibex_files_match_record']}/40).\n")

w("## 3. Commands (step 3)\n")
w("Each laptop psmc-rs command is the Ibex provenance record's `key.command` with only the placeholders replaced: the binary (`<repo>/target/release/psmc-rs` → the laptop build), the input (`<runs>/E1/sims/...` → the identical laptop genome; NA12878 → this checkout's file) and the two outputs (→ `xplat/runs/...`). Every option is unchanged, including `--threads 1`, `--mstep-iters 1000`, `-N 25`, the pattern, `--t-max 15`, `--tr-ratio 5`, `--smooth-lambda 0`, `--bin-size 100`, `--mu`, `--gen-years` and `--no-progress` (`jobs.json` lists both commands; `records/` holds each run's command, checksums, exit status and times). All 41 exited 0.\n")
w("Example (E1 constant rep01):\n")
w("```")
w("psmc-rs <laptop>/E1/sims/constant/rep01/genome.psmcfa <out>/rep01.rs.json 25 --pattern 4+25*2+4+6 --t-max 15 --tr-ratio 5 \\")
w("    --smooth-lambda 0 --mstep-iters 1000 --bin-size 100 --mu 2.5e-08 --gen-years 25 --threads 1 --no-progress --psmc-out <out>/rep01.rs.psmc")
w("```\n")

w("## 4. Results\n")
w("### 4.1 Byte identity\n")
w(f"- **psmc-rs, laptop vs Ibex: 0/41 byte-identical** (JSON {S['rs']['json_bytes_identical']}/41; JSON with its `simd` label set aside {S['rs']['json_identical_except_simd_label']}/41; `.psmc` {S['rs']['psmc_bytes_identical']}/41).")
w(f"- **C psmc, laptop build vs Ibex build: 0/40 byte-identical** (`.psmc` {S['c']['psmc_bytes_identical']}/40).")
if ctl199:
    w(f"- **Control, laptop rustc 1.99.0 vs laptop rustc 1.97.1: {sum(r['json_bytes_identical'] == 'True' for r in ctl199)}/{len(ctl199)} JSON and {sum(r['psmc_bytes_identical'] == 'True' for r in ctl199)}/{len(ctl199)} `.psmc` byte-identical.**")
if rep:
    w(f"- **Control, the same laptop binary run twice** ({', '.join(r['model'] + ' rep' + r['replicate'].zfill(2) for r in rep)}): {sum(r['json_bytes_identical'] == 'True' for r in rep)}/{len(rep)} byte-identical (JSON and `.psmc`).")
w("")
w("### 4.2 Where the runs part\n")
w(f"- **psmc-rs.** The log-likelihood after 0 iterations (the E-step at the starting parameters) is bit-identical in {S['rs']['loglike_iter0_identical']}/41 fits; in the other (E1 expansion rep05) it differs by one unit in the last place ({f(next(r for r in rs1 if r['model']=='expansion' and r['replicate']=='5')['loglike_iter0_rel'])} relative). The first log-likelihood that differs is that after iteration {', '.join(f"{k} ({v} fit{'s' if v != 1 else ''})" for k, v in sorted(S['rs']['loglike_first_diff_iter'].items(), key=lambda kv: int(kv[0])))}, at the level of the last digits; the first differing `.psmc` line is mostly the M-step's evaluation count (`IT`) of iterations 1 to 8 ({', '.join(f'{k}: {v}' for k, v in sorted(S['rs']['psmc_first_diff'].items(), key=lambda kv: -kv[1]))}). Rust does not contract floating-point operations, and per SN1 section 8 (and `src/hmm/simd.rs`) the NEON and the AVX2+FMA kernels fuse the same multiply-adds and no sum changes its order with the vector width, which the identical iteration-0 log-likelihoods bear out; what remains to differ is the system math library (Apple libm vs glibc, whose `exp`/`log` can round differently in the last bit) and any other architecture-specific code path. This run does not separate them. The M-step, stopping on a flat optimum, turns the last-digit differences into the parameter differences below.")
w(f"- **C psmc.** In all 40 fits the first differing line is the `IT` count of iteration 1's Hooke–Jeeves M-step ({', '.join(f'{k}: {v}' for k, v in S['c']['psmc_first_diff'].items())}); iteration 1's LK and QD-before lines are identical in {S['c']['round1_lk_and_qd_before_identical']}/40 (the E-step at the starting parameters agrees to C's printed precision, 6 decimals), and from there the evaluation counts differ in a median {int(med(c, 'mstep_counts_differ'))} of 25 iterations. The two C builds differ in compiler, optimisation level, FP contraction (the laptop binary fuses multiply-adds in the E-step and the M-step objective; Ibex's does not) and math library, which this comparison cannot separate.\n")

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
ctl_all_same = bool(ctl199) and all(r["json_bytes_identical"] == "True" for r in ctl199) and len(ctl199) == 41
show199 = bool(r199_1) and not ctl_all_same
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
w(f"- C psmc's largest parameter difference (0.21) and d (0.10) are expansion rep05's first λ, which runs away out of the window in both builds ({f(c_e5['param_max_rel'])} relative, {f(c_e5['dlog10ne_max_all_intervals'])} in log10; inside the window that fit agrees to {f(c_e5['dlog10ne_max_window'])}). Without that fit: largest parameter difference {f(mx(c_noexp5, 'param_max_rel'))}, d {f(mx(c_noexp5, 'dlog10ne_max_all_intervals'))}. (psmc-rs bounds every free λ at 1e4, so the same λ stops at the bound on both platforms; on the way there, at iteration 14, it differed by 8.5e-4 relative between them.)")
prm = rows("params_rel_per_fit.csv")
for comp, label in (("psmc-rs laptop vs Ibex", "psmc-rs"), ("C psmc laptop vs Ibex", "C psmc")):
    rr = [r for r in prm if r["comparison"] == comp and r["exp"] == "E1"]
    lam = {i: max(float(r[f"lam{i}"]) for r in rr) for i in range(28)}
    top = sorted(lam, key=lam.get, reverse=True)
    rest = [i for i in range(28) if i not in top[:3]]
    w(f"- Each λ, {label} (largest relative difference over the 40 fits; `params_rel_per_fit.csv`): "
      + ", ".join(f"λ{i} {f(lam[i])}" for i in top[:3]) + f"; the other 25 at most {f(max(lam[i] for i in rest))}.")
bx = Counter(r["param_max_rel_name"] for r in rs1 if r["model"] in ("bottleneck", "expansion"))
cz = Counter(r["param_max_rel_name"] for r in rs1 if r["model"] in ("constant", "zigzag"))
w(f"- psmc-rs's largest relative parameter difference is λ1 or λ2 (the recent intervals after the first λ) in {bx['lam1'] + bx['lam2']}/20 bottleneck and expansion fits; in the constant and zigzag fits it is the oldest λ (λ27) in {cz['lam27']}/20, t_max in {cz['t_max']}, another λ in the rest. Its largest d, {f(mx(rs1, 'dlog10ne_max_all_intervals'))}, is bottleneck rep10's first λ, outside the window (inside: {f(next(r for r in rs1 if r['model']=='bottleneck' and r['replicate']=='10')['dlog10ne_max_window'])}).\n")

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
w(f"- psmc-rs's cross-platform window RMSE is below its RMSE against C psmc in **{sc['rs_xplat_rmse_below_rs_vs_c']}/40** replicates; ratio (rs vs C) / (rs cross-platform), min / median / max: {ratio('ratio_rmse_rsvsc_over_rs_xplat')}.")
w(f"- psmc-rs's cross-platform RMSE is below C psmc's cross-build RMSE in **{sc['rs_xplat_rmse_below_c_xplat']}/40**; ratio (C xplat) / (rs xplat): {ratio('ratio_rmse_c_xplat_over_rs_xplat')}.")
w(f"- C psmc's cross-build RMSE is below the psmc-rs-vs-C RMSE in {sc['c_xplat_rmse_below_rs_vs_c']}/40; ratio (rs vs C) / (C xplat): {ratio('ratio_rmse_rsvsc_over_c_xplat')} (the noise floor of B7: psmc-rs vs C is a few times C vs C).")
w(f"- Log-likelihood, worst iteration (median): psmc-rs cross-platform {f(S['side_by_side']['loglike_rel_max_rs_laptop_vs_ibex']['median'])}, C cross-build {f(S['side_by_side']['loglike_rel_max_c_laptop_vs_ibex']['median'])}, psmc-rs vs C {f(S['side_by_side']['loglike_rel_max_rs_vs_c_ibex']['median'])}.\n")

w("## 5. Pre-registered tiers\n")
w("Tiers as defined in the review's `xplatform.py`, unchanged: **A** output bytes identical; **B** p ≤ 1e-6, d ≤ 1e-6 and final log-likelihood ≤ 1e-8 (relative); **C** d ≤ 1e-3 and 50·d ≤ the reference psmc-rs-vs-C RMSE; otherwise **fail**. Bytes for A: psmc-rs's JSON (as for the bootstrap replicates of the pre-registered script), C psmc's `.psmc`.\n")
w("The pre-registration leaves the tier-C reference unspecified (review B2). Primary: the per-model mean psmc-rs-vs-C RMSE of the committed `E1_table1.csv` (`rmse_rs_vs_c_per_interval_mean`, interval by interval after the B1 fix: constant "
  f"{f(S['references']['model_mean_primary']['constant'])}, bottleneck {f(S['references']['model_mean_primary']['bottleneck'])}, expansion {f(S['references']['model_mean_primary']['expansion'])}, zigzag {f(S['references']['model_mean_primary']['zigzag'])}); for NA12878 the mean over all 40 E1 replicates ({f(S['references']['all40_mean'])}), as the pre-registered script did for E2. These are smaller (stricter) than the pre-B1 references the review used.\n")
w("| tier-C reference | psmc-rs E1 (40) | psmc-rs NA12878 | C psmc E1 (40) |" + (" psmc-rs rustc 1.99.0 E1 (40) |" if show199 else ""))
w("|---|---|---|---|" + ("---|" if show199 else ""))
for k, label in (("tier", "**primary**: per-model mean (E2: all-40 mean)"),
                 ("tier_vs_pair_rmse", "the same replicate's rs-vs-C RMSE (NA12878: its own, 9.7e-6)"),
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
p_ok = sum(float(r["param_max_rel"]) <= 1e-6 for r in rs1)
d_ok = sum(float(r["dlog10ne_max_all_intervals"]) <= 1e-6 for r in rs1)
w(f"- **psmc-rs (primary): B {Counter(r['tier'] for r in rs1)['B']}, C {Counter(r['tier'] for r in rs1)['C']}, fail {len(fails)} of 40; NA12878: {na['tier']}.** Every one of the {40 - Counter(r['tier'] for r in rs1)['B']} fits that miss B exceeds p ≤ 1e-6 ({sum(float(r['dlog10ne_max_all_intervals']) <= 1e-6 for r in rs1 if r['tier'] != 'B')} on p alone, {sum(float(r['dlog10ne_max_all_intervals']) > 1e-6 for r in rs1 if r['tier'] != 'B')} also on d); the log-likelihood test passes in 40/40 (p ≤ 1e-6 in {p_ok}/40, d ≤ 1e-6 in {d_ok}/40). By model, B: constant {sum(r['tier']=='B' for r in rs1 if r['model']=='constant')}/10, bottleneck {sum(r['tier']=='B' for r in rs1 if r['model']=='bottleneck')}/10, expansion {sum(r['tier']=='B' for r in rs1 if r['model']=='expansion')}/10, zigzag {sum(r['tier']=='B' for r in rs1 if r['model']=='zigzag')}/10.")
w("- The five fails (reference / d needs ≥ 50): " + "; ".join(
    f"{r['model']} rep{int(r['replicate']):02d} (p {f(r['param_max_rel'])} on {r['param_max_rel_name']}, d {f(r['dlog10ne_max_all_intervals'])}, ratio {float(r['ratio_ref_over_d']):.0f})"
    for r in fails) + ". Bottleneck rep10's d is its out-of-window first λ; over the window it would be C.")
w(f"- **C psmc: fail 40/40** under every reference considered except the review's superseded pre-B1 one (C 1). It fails B on both p (≥ {f(S['c']['E1']['param_max_rel']['min'])}) and d (≥ {f(S['c']['E1']['dlog10ne_max_all_intervals']['min'])}), and C because 50·d exceeds the reference (reference/d at most {f(max(float(r['ratio_ref_over_d']) for r in c))}). Its log-likelihoods would pass B ({f(S['c']['E1']['loglike_rel_final']['max'])} at most). This reproduces the review's B7 for the 40 E1 fits (identical numbers; there: A 0, B 0, C 1, fail 43 of 44 with the pre-B1 reference).")
w("- Note on the tiers' structure: with the corrected references, tier C's d-test (d ≤ reference/50: 2.1e-7 constant, 1.4e-6 bottleneck, 1.1e-6 expansion, 3.8e-7 zigzag) is stricter than tier B's d ≤ 1e-6 for the constant and zigzag models, so the tiers are not nested: zigzag rep01 and rep08 are B although their d would fail the C test.\n")

if ctl199:
    w("## 6. Control: the laptop build with Ibex's rustc (1.99.0)\n")
    same = sum(r["json_bytes_identical"] == "True" for r in ctl199)
    w(f"The same sources built on the laptop by rustc 1.99.0 (b940084d7eb6, the commit of Ibex's rustc; LLVM 23.1.1) gave {same}/{len(ctl199)} fits byte-identical to the rustc 1.97.1 build's (JSON and `.psmc`). "
      + ("The rustc version therefore does not change psmc-rs's output on the laptop, and the laptop-vs-Ibex differences above are those of the platform (CPU architecture and kernels, operating system and its math library), not of the compiler version." if same == len(ctl199) else
         "The rustc version changes the output for some fits; the comparison with Ibex at the same rustc is in `rs_laptop_rustc1.99_vs_ibex.csv` and in the last columns of the tables above."))
    w("")

w("## 7. Deviations from the pre-registration\n")
w("1. **Tier A for psmc-rs** is tested on the JSON, as pre-registered, but the JSON records the kernels that ran (`simd`: `neon` vs `avx2+fma`), so it can never be byte-identical across these two platforms. Also testing the JSON without that label, and the `.psmc` file, changes nothing: no fit is identical either way.")
w("2. **Tier-C reference.** Unspecified in the pre-registration; the primary choice (per-model mean, matched intervals) is the corrected version (B1) of what the review's script used. All alternatives are reported (table in section 5).")
w("3. **d over the window only** (excluding an out-of-window first λ) is reported as a labelled deviation, never as the result; it moves psmc-rs bottleneck rep10 from fail to C and changes nothing for C psmc.")
w("4. **Compiler version.** The laptop build used the default toolchain (rustc 1.97.1), not Ibex's 1.99.0; the rustc 1.99.0 control (section 6) addresses it.")
w("5. **NA12878 window and reference.** Its window is E2's committed one; its tier-C reference is the all-40 E1 mean (the pre-registered rule for E2 fits).\n")

w("## 8. Caveats\n")
w("- **Threads:** identical on both platforms (psmc-rs `--threads 1` as recorded; C psmc is single-threaded). The determinism control (one binary, two runs, byte-identical) shows the laptop differences are not run-to-run noise.")
w("- **Laptop C psmc** is the frozen `.production` run's binary and fits; its compiler and flags are not in its provenance. The evidence above (machine code identical to the -O1 builds; Apple clang 21.0.0 from a sibling build's DWARF; FP contraction on) is circumstantial. Not done: a laptop C build with `-ffp-contract=off` (about 16 min per fit) to separate FMA contraction from the compiler and libm.")
w("- **Ibex CPU and OS** for the E1/E2 jobs are not in their records; the fits' `simd: avx2+fma` confirms an AVX2+FMA CPU.")
w("- **psmc-rs source:** now the same on both platforms (`src` 4c0d777), which removes the review's m21 confound (the earlier laptop psmc-rs was df00bf2e). The laptop's C fits and simulated genomes come from that earlier `.production` run (they do not depend on psmc-rs).")
w("- **Scope:** E1's 40 fits of each tool and NA12878's psmc-rs fit. Not redone on the laptop: NA12878's C fit, the turtle genomes, bootstraps, E4–E7.")
w("- **Concurrent work:** during this session another session moved HEAD from 980bab8 to 2b9a3d8 (removing E2's Papuan outputs), edited `.gitignore` and added `experiment/runs/E7/inputs/`; none of that touches the Rust sources (still 4c0d777 / 7ff6ee3 / b552a2a, clean) or any file used here.\n")

w("## 9. Files (`xplat/`)\n")
for name, desc in (
    ("inputs_E1_genomes.csv", "step 1, one row per genome (40)"),
    ("inputs_E1_chromosomes.csv", "step 1, one row per chromosome (400)"),
    ("inputs.json", "step 1 counts, NA12878"),
    ("rs_laptop_vs_ibex.csv", "psmc-rs laptop (rustc 1.97.1) vs Ibex, 41 rows, with tiers"),
    ("c_laptop_vs_ibex.csv", "C psmc laptop vs Ibex, 40 rows, with tiers"),
    ("rs_vs_c_ibex.csv", "psmc-rs vs C psmc on Ibex, recomputed with the same code (matches E1_rs_vs_c.csv to %s)" % f(S["repro_committed_rs_vs_c_max_abs_error"])),
    ("side_by_side_E1.csv", "the three agreements per E1 replicate, and their ratios"),
    ("params_rel_per_fit.csv", "relative difference of θ, ρ, t_max and each of the 28 free λ, per fit (psmc-rs and C psmc, laptop vs Ibex)"),
    ("rs_laptop_rustc1.99_vs_ibex.csv", "control: laptop rustc 1.99.0 build vs Ibex"),
    ("control_rustc1.99_vs_rustc1.97_laptop.csv", "control: the two laptop builds"),
    ("control_repeat_laptop.csv", "control: one laptop binary, run twice"),
    ("summary.json", "every number above"),
    ("toolchain.txt", "laptop machine, rustc, binary checksums, C psmc binary forensics"),
    ("jobs.json, jobs_rustc1.99.json, jobs_repeat.json", "the Ibex and the laptop command of each job"),
    ("records/, records-rustc1.99/, records-repeat/", "one record per laptop run: command, input/output checksums, exit status, times"),
    ("runs/, runs-rustc1.99/, runs-repeat/", "the laptop fits (JSON and .psmc)"),
    ("bin/", "the laptop psmc-rs binaries that ran"),
    ("build/", "cargo build logs"),
):
    w(f"- `{name}`: {desc}")
(X / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out)[:3000])
