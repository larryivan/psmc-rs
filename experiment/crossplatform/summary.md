# Cross-platform comparison: laptop vs Ibex (data for Supplementary Note 1, section 9)

Computed on 2026-10-25 by the scripts of `scripts/` (`make_jobs.py`, `run_job.py`, `toolchain.py`, `inputs.py`, `compare.py`, `reports.py`, `make_summary.py`; README.md gives the order). The tiers of section 5 are those `scripts/tiers_2026-10-07.py` defined on 2026-10-07, before these runs.

The Ibex fits compared are the production's, committed at cbab860c: its psmc-rs fits ran at 4b2c636f with a binary built from `src` dcf29ca, the sources the laptop built; its C psmc fits come from earlier jobs of the same C psmc binary, which did not change.

## Headline

- **Inputs identical:** the laptop's 40 E1 genomes are byte-identical to Ibex's (psmcfa, VCF.gz and index: 40/40, 40/40, 40/40; 400/400 per-chromosome checksums and counts); NA12878's input too.
- **psmc-rs, same sources, laptop (arm64 NEON, macOS) vs Ibex (x86-64 AVX2+FMA, Linux): 0/41 outputs byte-identical.** Median (max) over the 40 E1 fits: θ 1.2e-07 (1.9e-06) relative, any parameter 5.3e-07 (2.1e-05), log10 Ne per interval 2.1e-07 (8.2e-06), window RMSE 8.8e-08 (2.4e-06), final log-likelihood 1.9e-12 (1.5e-11). NA12878: parameters 1.1e-06, log10 Ne 2.9e-07, window RMSE 1.5e-07, log-likelihood 4.3e-12.
- **Pre-registered tiers, psmc-rs:** B 27, C 10, **fail 3** of 40 (NA12878: C). **C psmc (laptop Apple clang build vs Ibex gcc -O3 build): fail 40 of 40.**
- **Side by side (window RMSE, per replicate):** psmc-rs's cross-platform difference is 383 times (median; at least 39) smaller than its difference from C psmc, and 124 times (median; at least 10) smaller than C psmc's own cross-build difference, which is in turn 2.6 times smaller than psmc-rs vs C (median; C's cross-build RMSE is the larger in 7/40).
- **Controls:** the same sources built on the laptop by Ibex's rustc (1.99.0) give output byte-identical to the rustc 1.97.1 build's in 41/41 fits, and one binary run twice gives identical bytes (2/2): the differences are the platform's (architecture, kernels, math library), not the compiler version's or run-to-run noise.
- **The analysis code:** E1, E2, E4 and E5's reports, run on the laptop on the committed fits with the code that wrote the committed tables on Ibex, write 16 of the 38 tables byte for byte; the others agree to a relative 2.5e-13 in every value of magnitude 1e-3 or more (section 4.6).

## 1. Machines and toolchains

| | Laptop | Ibex (production) |
|---|---|---|
| CPU | Apple M1 Pro (arm64; 8 cores: 6 performance, 2 efficiency; 16 GB) | AMD EPYC 7702 (x86-64, Zen 2), as SN1 and E3's environment table give for the cluster's nodes; the fit records name the hosts (psmc-rs, E1 and NA12878: cn513-11-l, SLURM job 54256959, 41 fits; C psmc, E1: cn514-11-r, SLURM job 53386770, 40 fits) but not their CPU model |
| OS | macOS 27.0 (26A428), Darwin 27.0.0, Apple libm | Linux x86-64 (E3's nodes: 5.14.0-427.20.1.el9_4, glibc 2.34; not recorded for the E1/E2 jobs) |
| psmc-rs sources | `src` dcf29ca, `Cargo.toml` c0be5cf, `Cargo.lock` ae10354, clean (HEAD ac4fed5 for both builds) | the same three ids, clean (every E1/E2 fit-rs provenance record) |
| rustc | 1.97.1 (8bab26f4f 2026-07-14), LLVM 22.1.6, aarch64-apple-darwin — the default toolchain of `cargo build --release --locked`; control build: 1.99.0 (b940084d7 2026-09-28), LLVM 23.1.1 | 1.99.0 (b940084d7eb6), x86_64-unknown-linux-gnu |
| build flags | release profile, no RUSTFLAGS / CARGO_*RUSTFLAGS, `.cargo/config.toml`: none; `rustc --print cfg` has `neon` | none (`rustflags` empty, `target_features: none`) |
| HMM kernels (`simd` in the fit JSON) | `neon` (41/41 fits) | `avx2+fma` (41/41 fits) |
| psmc-rs binary | `psmc-rs 0.2.0`, sha256 44d073c2… (rustc 1.97.1), 698e0976… (rustc 1.99.0); without the Mach-O UUID and the code signature, 0fd9e912… and 79eeea94… | `psmc-rs 0.2.0`, sha256 1d141b81… (also the binary of E3's timings) |
| C psmc | 0.6.5-r74-dirty, sha256 c40db274… (the frozen `.production` run, fits of 2026-10-06). Compiler and CFLAGS are in no provenance record. Its machine code (`__text`) is identical to the `psmc_O1` and `psmc_O` binaries beside it in `experiment/.tools/psmc_c` (so: -O1); the DWARF of the sibling ASan build names Apple clang version 21.0.0 (clang-2100.3.34.2); it contains 23 fused multiply-add instructions (in `psmc_update_hmm`, `hmm_forward`, `hmm_backward`, `hmm_expect`, `hmm_Q`, `hmm_Q0`, …), i.e. clang's default FP contraction was on | 0.6.5-r74-dirty, sha256 d5e0e160…, gcc 11.4.1 (Red Hat 11.4.1-3), `-g -Wall -O3 -march=x86-64-v3 -ffp-contract=off -Wno-unused-function` (fits of 2026-10-07) |
| threads | psmc-rs `--threads 1` (as recorded on Ibex); C psmc single-threaded | the same |

The laptop binaries' whole-file sha256 depends on the target directory they were built in: rebuilt from the same sources by the same toolchain in another target directory, each differed in its Mach-O UUID (LC_UUID), and so in its ad-hoc code signature, and nowhere else. The sha256 with those bytes zeroed (`toolchain.txt`; `scripts/toolchain.py --digest <binary>` computes it) is the same for such a rebuild. Both still depend on the source paths a binary holds in its panic messages, under the builder's home directory (`~/.cargo`, `~/.rustup`), so another user's or machine's build is not expected to reproduce either.

psmc-rs wall time per E1 fit (not a controlled timing): laptop 40–48 s (4 jobs at once, `nice 19`); Ibex 52–77 s (the production's fits shared their node). NA12878: laptop 235 s, Ibex 394 s.

## 2. Inputs (step 1)

- **E1 genomes: 40/40 byte-identical.** The SHA-256 of every laptop `genome.psmcfa` and `genome.vcf.gz` equals the one in the committed Ibex truth record (40/40 and 40/40) and in the Ibex sim provenance record (psmcfa 40/40, vcf.gz 40/40, tabix index 40/40).
- **Per chromosome: 400/400 match.** Recomputed from the laptop files, each chromosome's heterozygous-position checksum (SHA-256 of the 0-based positions as little-endian int64, read from the VCF) equals the Ibex truth record's `het_positions_sha256`, and so do its counts (bins and heterozygous bins from the psmcfa; heterozygous and derived homozygous sites and the last heterozygous position from the VCF): 400/400.
- **Fit inputs.** The psmcfa checksum each Ibex fit recorded as its input equals the laptop file's (40/40 psmc-rs, 40/40 C psmc). The laptop C fits read the same files: their records' input checksums equal them (40/40), their outputs still match their records (40/40) and the copies in `laptop_c/` (40/40), and their `.psmc` headers give the same n_seqs, sum_L and sum_n as Ibex's (40/40). The sim records' command, seeds and settings agree (40/40), and so do the truth records' chromosome blocks (40/40).
- **NA12878** (`experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz`, sha256 a9501bcc5082…) is the file both Ibex NA12878 fits read (yes).
- The production's Ibex fits are the files their provenance records name (psmc-rs 41/41, C psmc 40/40).

## 3. Commands (step 3)

Each laptop psmc-rs command is the Ibex provenance record's `key.command` with only the placeholders replaced: the binary (`<repo>/target/release/psmc-rs` → the laptop build), the input (`<runs>/E1/sims/...` → the identical laptop genome; NA12878 → this checkout's file) and the two outputs (→ `runs/...`). Every option is unchanged, including `--threads 1`, `--mstep-iters 1000`, `-N 25`, the pattern, `--t-max 15`, `--tr-ratio 5`, `--smooth-lambda 0`, `--bin-size 100`, `--mu`, `--gen-years` and `--no-progress` (`jobs.json` lists both commands; `records/` holds each run's command, checksums, exit status and times). All 41 exited 0.

Example (E1 constant rep01):

```
psmc-rs <laptop>/E1/sims/constant/rep01/genome.psmcfa <out>/rep01.rs.json 25 --pattern 4+25*2+4+6 --t-max 15 --tr-ratio 5 \
    --smooth-lambda 0 --mstep-iters 1000 --bin-size 100 --mu 2.5e-08 --gen-years 25 --threads 1 --no-progress --psmc-out <out>/rep01.rs.psmc
```

## 4. Results

### 4.1 Byte identity

- **psmc-rs, laptop vs Ibex: 0/41 byte-identical** (JSON 0/41; JSON with its `simd` label set aside 0/41; `.psmc` 0/41).
- **C psmc, laptop build vs Ibex build: 0/40 byte-identical** (`.psmc`).
- **Control, laptop rustc 1.99.0 vs laptop rustc 1.97.1: 41/41 JSON and 41/41 `.psmc` byte-identical.**
- **Control, the same laptop binary run twice** (constant rep01, bottleneck rep06): 2/2 byte-identical (JSON and `.psmc`).

### 4.2 Where the runs part

- **psmc-rs.** The log-likelihood after 0 iterations (the E-step at the starting parameters) is bit-identical in 40/41 fits; in the other, E1 expansion rep05, it differs by one unit in the last place (2.0e-16 relative). The first log-likelihood that differs is that after iteration 0 (1 fit), 1 (34 fits), 2 (5 fits), 3 (1 fit), at the level of the last digits; the first differing `.psmc` line is mostly the M-step's evaluation count (`IT`) of iterations 1 to 5 (round 1 IT: 17, round 2 IT: 13, round 3 IT: 5, round 1 RI: 3, round 3 RI: 1, round 4 IT: 1, round 5 IT: 1). Rust does not contract floating-point operations, and per SN1 section 8 (and `src/hmm/simd.rs`) the NEON and the AVX2+FMA kernels fuse the same multiply-adds and no sum changes its order with the vector width, which the identical iteration-0 log-likelihoods bear out; what remains to differ is the system math library (Apple libm vs glibc, whose `exp`/`log` can round differently in the last bit) and any other architecture-specific code path. This run does not separate them. The M-step, stopping on a flat optimum, turns the last-digit differences into the parameter differences below.
- **C psmc.** In all 40 fits the first differing line is the `IT` count of iteration 1's Hooke–Jeeves M-step (round 1 IT: 40); iteration 1's LK and QD-before lines are identical in 40/40 (the E-step at the starting parameters agrees to C's printed precision, 6 decimals), and from there the evaluation counts differ in a median 25 of 25 iterations. The two C builds differ in compiler, optimisation level, FP contraction (the laptop binary fuses multiply-adds in the E-step and the M-step objective; Ibex's does not) and math library, which this comparison cannot separate.

### 4.3 Agreement (min / median / max over the 40 E1 fits; NA12878 single fit)

Relative differences are |a − b| / |b| with Ibex as b. Δlog10 Ne is compared interval by interval (`metrics.interval_diff`; Ne_k = λ_k·N0, N0 ∝ θ): 'all intervals' is the pre-registered d (all 64 intervals, the out-of-window first λ included); 'window' uses the committed informative windows of `E1_rs_vs_c.csv` (NA12878: `E2_rs_vs_c.csv`), weighted by log time as in E1. C psmc's parameters come from its `PA` lines (9 decimals) and its last log-likelihood is iteration 24's (C prints none for the final parameters); psmc-rs's from its JSON (full precision).

| quantity | psmc-rs laptop vs Ibex | C psmc laptop vs Ibex | psmc-rs vs C psmc (both Ibex) | NA12878 psmc-rs laptop vs Ibex |
|---|---|---|---|---|
| θ | 9.1e-09 / 1.2e-07 / 1.9e-06 | 1.5e-07 / 5.8e-06 / 5.3e-05 | 8.5e-07 / 6.3e-05 / 3.8e-04 | 4.6e-07 |
| ρ | 3.3e-09 / 1.2e-07 / 1.8e-06 | 1.6e-07 / 5.8e-06 / 5.3e-05 | 1.3e-06 / 5.4e-05 / 3.6e-04 | 4.6e-07 |
| t_max | 1.0e-09 / 1.8e-07 / 2.9e-06 | 9.1e-08 / 9.5e-06 / 8.1e-05 | 5.1e-06 / 7.7e-05 / 5.5e-04 | 7.5e-07 |
| largest free λ | 3.1e-08 / 5.3e-07 / 2.1e-05 | 3.0e-05 / 1.1e-04 / 2.1e-01 | 6.0e-05 / 2.2e-04 / 9.9e-01 | 1.1e-06 |
| median free λ | 1.6e-08 / 1.3e-07 / 2.0e-06 | 4.1e-06 / 8.7e-06 / 5.3e-05 | 6.7e-06 / 6.2e-05 / 4.0e-04 | 5.1e-07 |
| largest of all (pre-registered p) | 3.1e-08 / 5.3e-07 / 2.1e-05 | 3.0e-05 / 1.1e-04 / 2.1e-01 | 6.0e-05 / 2.2e-04 / 9.9e-01 | 1.1e-06 |
| max \|Δlog10 Ne\|, all intervals (pre-registered d) | 1.3e-08 / 2.1e-07 / 8.2e-06 | 1.4e-05 / 4.9e-05 / 1.0e-01 | 1.0e-05 / 7.8e-05 / 2.0e+00 | 2.9e-07 |
| max \|Δlog10 Ne\|, window | 1.3e-08 / 2.1e-07 / 8.2e-06 | 5.6e-06 / 2.0e-05 / 8.1e-05 | 9.0e-06 / 5.5e-05 / 6.6e-04 | 2.9e-07 |
| RMSE Δlog10 Ne, window | 4.8e-09 / 8.8e-08 / 2.4e-06 | 3.6e-06 / 8.5e-06 / 2.4e-05 | 3.7e-06 / 2.5e-05 / 2.0e-04 | 1.5e-07 |
| max \|Δlog10 t_k\| (time scale) | 6.4e-09 / 3.9e-08 / 5.7e-07 | 1.2e-07 / 1.9e-06 / 1.6e-05 | 6.2e-06 / 2.6e-05 / 1.2e-04 | 1.4e-07 |
| final log-likelihood (pre-registered) | 3.6e-15 / 1.9e-12 / 1.5e-11 | 4.9e-12 / 1.7e-10 / 1.2e-09 | 1.5e-10 / 3.3e-09 / 1.3e-08 | 4.3e-12 |
| log-likelihood, worst iteration | 3.3e-12 / 1.7e-11 / 7.1e-11 | 6.0e-10 / 1.6e-09 / 6.3e-09 | 1.0e-09 / 8.4e-09 / 3.5e-08 | 3.5e-11 |
| log-likelihood, iteration 0 | 0 / 0 / 2.0e-16 | 0 / 0 / 0 | 3.1e-14 / 2.5e-13 / 7.0e-13 | 0 |

- C psmc's largest parameter difference (0.21) and d (0.10) are expansion rep05's first λ, which runs away out of the window in both builds (8.55e+05 on the laptop, 1.08e+06 on Ibex: 2.1e-01 relative, 1.0e-01 in log10; inside the window that fit agrees to 8.1e-05). Without that fit: largest parameter difference 5.4e-04, d 2.4e-04. (psmc-rs bounds every free λ at 1e4, so the same λ stops at the bound on both platforms, at 9999.99999998957 on the laptop and 9999.99999999 on Ibex; on the way there, at iteration 14, it differed by 3.0e-03 relative between them.) The same replicate holds psmc-rs's largest differences: its iteration-0 log-likelihood is the one that differs, and its fit differs by 2.1e-05 on λ1, by a d of 8.2e-06 on λ1, inside the window, and by an RMSE of 2.4e-06 over the window, the largest of the 40.
- Each λ, psmc-rs (largest relative difference over the 40 fits; `params_rel_per_fit.csv`): λ1 2.1e-05, λ0 7.8e-06, λ2 4.7e-06; the other 25 at most 3.7e-06.
- Each λ, C psmc (largest relative difference over the 40 fits; `params_rel_per_fit.csv`): λ0 2.1e-01, λ27 5.4e-04, λ1 2.4e-04; the other 25 at most 1.9e-04.
- psmc-rs's largest relative parameter difference is, in the 20 bottleneck and expansion fits: λ1 12, λ0 4, λ2 3, λ4 1; in the 20 constant and zigzag fits: t_max 5, λ2 4, λ1 3, λ27 3, λ0 2, λ11 1, λ23 1, λ4 1. Its largest d, 8.2e-06, is expansion rep05's λ1, inside the window.

### 4.4 Per model (median [max] over 10 replicates)

| model | RMSE window: rs xplat | C xplat | rs vs C (Ibex) | max window: rs xplat | C xplat | rs vs C | d all: rs xplat | C xplat | rs vs C |
|---|---|---|---|---|---|---|---|---|---|
| constant | 1.9e-08 [1.4e-07] | 8.0e-06 [1.2e-05] | 7.8e-06 [3.5e-05] | 4.1e-08 [2.2e-07] | 1.7e-05 [2.5e-05] | 1.5e-05 [9.2e-05] | 5.2e-08 [2.2e-07] | 2.7e-05 [7.7e-05] | 2.5e-05 [9.2e-05] |
| bottleneck | 1.2e-07 [2.7e-07] | 9.9e-06 [2.2e-05] | 3.9e-05 [1.9e-04] | 4.1e-07 [9.1e-07] | 2.9e-05 [7.9e-05] | 1.1e-04 [6.6e-04] | 4.5e-07 [3.4e-06] | 4.7e-05 [1.4e-04] | 1.4e-04 [6.6e-04] |
| expansion | 1.2e-07 [2.4e-06] | 1.5e-05 [2.4e-05] | 3.0e-05 [2.0e-04] | 3.7e-07 [8.2e-06] | 4.9e-05 [8.1e-05] | 7.2e-05 [4.8e-04] | 4.7e-07 [8.2e-06] | 5.2e-05 [1.0e-01] | 8.5e-05 [2.0e+00] |
| zigzag | 4.9e-08 [9.5e-08] | 7.7e-06 [1.6e-05] | 1.9e-05 [3.1e-05] | 9.8e-08 [2.4e-07] | 1.5e-05 [2.8e-05] | 4.2e-05 [6.6e-05] | 9.8e-08 [2.4e-07] | 7.4e-05 [2.4e-04] | 9.7e-05 [3.2e-04] |

### 4.5 Side by side (per E1 replicate; `side_by_side_E1.csv`)

- psmc-rs's cross-platform window RMSE is below its RMSE against C psmc in **40/40** replicates; ratio (rs vs C) / (rs cross-platform), min / median / max: 3.9e+01 / 3.8e+02 / 4.5e+03.
- psmc-rs's cross-platform RMSE is below C psmc's cross-build RMSE in **40/40**; ratio (C xplat) / (rs xplat): 1.0e+01 / 1.2e+02 / 9.8e+02.
- C psmc's cross-build RMSE is below the psmc-rs-vs-C RMSE in 33/40; ratio (rs vs C) / (C xplat): 4.5e-01 / 2.6e+00 / 1.9e+01 (the noise floor of B7: psmc-rs vs C is a few times C vs C).
- Log-likelihood, worst iteration (median): psmc-rs cross-platform 1.7e-11, C cross-build 1.6e-09, psmc-rs vs C 8.4e-09.

### 4.6 The tables, written again on the laptop (`reports.py`)

E1, E2, E4 and E5's reports (`python -m experiment.pipeline report`) ran on the laptop on a copy of the committed runs directory, its tables removed, and wrote the 38 tables again; the committed ones are those the production's reports wrote on Ibex (cn513-11-l, SLURM job 54256959), and each report's identity on the laptop (the digest of its code, Python's and the packages' versions) is that of its Ibex record (4/4). `reports_laptop_vs_ibex.csv` compares them cell by cell:

- 16/38 tables byte-identical, all 38 with the same rows and columns; 7716 of 295344 cells differ, none of them text (0).
- Every value of magnitude 1e-3 or more agrees to a relative 2.5e-13 at most; smaller values to 8.9e-16 absolute.
- The differences between psmc-rs and C psmc (196 values of the `*rs_vs_c*` columns of `E1_rs_vs_c.csv`, `E1_table1.csv`, `E2_rs_vs_c.csv`, of order 1e-5) agree to a relative 1.9e-11 (8.9e-16 absolute).
- The same code on the laptop recomputes the committed `E1_rs_vs_c.csv` (`compare.py`'s `rs_vs_c_ibex.csv`) to 8.9e-16 absolute.

## 5. Pre-registered tiers

Tiers as defined in the review's `xplatform.py`, unchanged: **A** output bytes identical; **B** p ≤ 1e-6, d ≤ 1e-6 and final log-likelihood ≤ 1e-8 (relative); **C** d ≤ 1e-3 and 50·d ≤ the reference psmc-rs-vs-C RMSE; otherwise **fail**. Bytes for A: psmc-rs's JSON (as for the bootstrap replicates of the pre-registered script), C psmc's `.psmc`.

The pre-registration leaves the tier-C reference unspecified (review B2). Primary: the per-model mean psmc-rs-vs-C RMSE of the committed `E1_table1.csv` (`rmse_rs_vs_c_per_interval_mean`, interval by interval after the B1 fix: constant 1.0e-05, bottleneck 7.0e-05, expansion 5.6e-05, zigzag 1.9e-05); for NA12878 the mean over all 40 E1 replicates (3.9e-05), as the pre-registered script did for E2. These are smaller (stricter) than the pre-B1 references the review used.

| tier-C reference | psmc-rs E1 (40) | psmc-rs NA12878 | C psmc E1 (40) |
|---|---|---|---|
| **primary**: per-model mean (E2: all-40 mean) | B 27, C 10, fail 3 | C 1 | fail 40 |
| the same replicate's rs-vs-C RMSE (NA12878: its own, 9.7e-06) | B 27, C 6, fail 7 | fail 1 | fail 40 |
| mean of all 40 E1 replicates | B 27, C 9, fail 4 | C 1 | fail 40 |
| smallest per-model mean (constant) | B 27, fail 13 | fail 1 | fail 40 |
| the review's superseded pre-B1 references (sampled curves; larger) | B 27, C 12, fail 1 | C 1 | C 1, fail 39 |
| *deviation*: d over the window's intervals only | B 27, C 11, fail 2 | C 1 | fail 40 |
| *deviation*: tier A ignores the JSON's `simd` label | B 27, C 10, fail 3 | C 1 | – |

- **psmc-rs (primary): B 27, C 10, fail 3 of 40; NA12878: C.** Every one of the 13 fits that miss B exceeds p ≤ 1e-6 (10 on p alone, 3 also on d; the parameter: λ1 8, λ0 4, λ2 1); the log-likelihood test passes in 40/40 (p ≤ 1e-6 in 27/40, d ≤ 1e-6 in 37/40). By model, B: constant 10/10, bottleneck 5/10, expansion 2/10, zigzag 10/10. NA12878 misses B on p (1.1e-06 on λ0; d 2.9e-07, reference/d 133).
- The 3 fails (reference / d needs ≥ 50): bottleneck rep10 (p 7.8e-06 on λ0, d 3.4e-06 on λ0, ratio 20; outside the window, and over the window it would be C); expansion rep05 (p 2.1e-05 on λ1, d 8.2e-06 on λ1, ratio 7); expansion rep09 (p 3.4e-06 on λ1, d 1.2e-06 on λ1, ratio 46).
- **C psmc: fail 40 of 40** under every reference considered except the review's superseded pre-B1 one (C 1, fail 39). It fails B on both p (≥ 3.0e-05) and d (≥ 1.4e-05), and C because 50·d exceeds the reference (reference/d at most 2.9e+00). Its log-likelihoods would pass B (1.2e-09 at most). Its fits are those of the review's B7 for these 40 genomes (there: A 0, B 0, C 1, fail 43 of 44 with the pre-B1 reference).
- Note on the tiers' structure: with the corrected references, tier C's d-test (d ≤ reference/50: 2.1e-07 constant, 1.4e-06 bottleneck, 1.1e-06 expansion, 3.8e-07 zigzag) is stricter than tier B's d ≤ 1e-6 for the constant and zigzag models, so the tiers are not nested: constant rep07 is B although its d would fail the C test.

## 6. Control: the laptop build with Ibex's rustc (1.99.0)

The same sources built on the laptop by rustc 1.99.0 (b940084d7 2026-09-28, the commit of Ibex's rustc, b940084d7eb6; LLVM 23.1.1) gave 41/41 fits byte-identical to the rustc 1.97.1 build's (JSON and `.psmc`). The rustc version therefore does not change psmc-rs's output on the laptop, and the laptop-vs-Ibex differences above are those of the platform (CPU architecture and kernels, operating system and its math library), not of the compiler version.

## 7. Deviations from the pre-registration

1. **Tier A for psmc-rs** is tested on the JSON, as pre-registered, but the JSON records the kernels that ran (`simd`: `neon` vs `avx2+fma`), so it can never be byte-identical across these two platforms. Also testing the JSON without that label, and the `.psmc` file, changes nothing: no fit is identical either way.
2. **Tier-C reference.** Unspecified in the pre-registration; the primary choice (per-model mean, matched intervals) is the corrected version (B1) of what the review's script used. All alternatives are reported (table in section 5).
3. **d over the window only** (excluding an out-of-window first λ) is reported as a labelled deviation, never as the result; it moves psmc-rs bottleneck rep10 from fail to C and changes nothing for C psmc.
4. **Compiler version.** The laptop build used the default toolchain (rustc 1.97.1), not Ibex's 1.99.0; the rustc 1.99.0 control (section 6) addresses it.
5. **NA12878 window and reference.** Its window is E2's committed one; its tier-C reference is the all-40 E1 mean (the pre-registered rule for E2 fits).

## 8. Caveats

- **Threads:** identical on both platforms (psmc-rs `--threads 1` as recorded; C psmc is single-threaded). The determinism control (one binary, two runs, byte-identical) shows the laptop differences are not run-to-run noise.
- **Laptop C psmc** is the frozen `.production` run's binary and fits; its compiler and flags are not in its provenance. The evidence above (machine code identical to the -O1 builds; Apple clang version 21.0.0 from a sibling build's DWARF; FP contraction on) is circumstantial. Not done: refitting the 40 genomes with a laptop C build with `-ffp-contract=off` (the laptop's C fits compared here took 15–25 min each), which would separate FMA contraction from the compiler and libm.
- **Ibex CPU and OS** for the E1/E2 jobs are not in their records; the fits' `simd: avx2+fma` confirms an AVX2+FMA CPU.
- **psmc-rs source:** the same on both platforms (`src` dcf29ca, the production's), which removes the review's m21 confound (the review's laptop psmc-rs was df00bf2e). The laptop's C fits and simulated genomes come from an earlier laptop run, `.production` (they do not depend on psmc-rs).
- **Scope:** E1's 40 fits of each tool and NA12878's psmc-rs fit, and the tables of E1, E2, E4 and E5. Not refitted on the laptop: NA12878's C fit, the turtle genomes, bootstraps, E4–E7.

## 9. Files (in this directory)

- `inputs_E1_genomes.csv`, `inputs_E1_chromosomes.csv`, `inputs.json`: step 1 (40 genomes, 400 chromosomes, NA12878)
- `rs_laptop_vs_ibex.csv`: psmc-rs laptop (rustc 1.97.1) vs Ibex, 41 rows, with tiers
- `c_laptop_vs_ibex.csv`: C psmc laptop vs Ibex, 40 rows, with tiers
- `rs_vs_c_ibex.csv`: psmc-rs vs C psmc on Ibex, recomputed with the same code (matches the committed `E1_rs_vs_c.csv` to 8.9e-16)
- `side_by_side_E1.csv`: the three agreements per E1 replicate, and their ratios
- `params_rel_per_fit.csv`: relative difference of θ, ρ, t_max and each of the 28 free λ, per fit
- `rs_laptop_rustc1.99_vs_ibex.csv`, `control_rustc1.99_vs_rustc1.97_laptop.csv`, `control_repeat_laptop.csv`: the controls
- `runs-rustc1.99/`, `runs-repeat/`: the controls' fits (41 by the rustc 1.99.0 build, byte-identical to `runs/`; 2 by the same binary run again); `jobs_rustc1.99.json`, `jobs_repeat.json`: their job lists; `records-rustc1.99/`, `records-repeat/`: their records
- `reports_laptop_vs_ibex.csv`, `reports.json`: the tables written again on the laptop, table by table (section 4.6)
- `summary.json`: every number above; `toolchain.txt`: the laptop, its two rustc builds, the binaries' checksums and the C psmc binary's forensics
- `jobs.json`: the Ibex and the laptop command of each job
- `runs/`: the laptop psmc-rs fits (JSON and `.psmc`); `records/`: one record per laptop run (command, checksums, exit status, times)
- `laptop_c/`: the laptop C psmc fits compared here, copied from the frozen laptop run `experiment/.production` (not in the repository)
- not kept: logs, the HTML reports, the binaries (`bin/`) and the build directories
