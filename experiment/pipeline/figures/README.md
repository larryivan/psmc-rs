# Figures

`experiment/pipeline/figures/` draws the manuscript's figures from the
experiments' tables. Each figure is one module with `draw(source)`; the
experiment `FIG` (`experiments/fig.py`) runs each as a runner job whose
inputs are the tables it reads, so a figure is redrawn exactly when a table,
the figure's code, its settings or the font change. Every figure is written
as `<runs>/<EXP>/figures/<name>.{pdf,svg,png}`: vector PDF and SVG
with text as text (TrueType, `fonttype 42`; SVG `svg.fonttype none`) and a
600-dpi PNG, without creation dates, so the same tables give the same bytes.

```bash
PY=experiment/.venv/bin/python
RUNS=/path/to/new/runs
$PY -m experiment.pipeline start --runs-dir $RUNS E1 FIG   # figures as runner jobs, after their tables
$PY -m experiment.pipeline.figures                         # or draw every figure whose tables exist
$PY -m experiment.pipeline.figures -j 4                    # the same, four figures at once (a process each)
$PY -m experiment.pipeline.figures fig2 figS1              # some of them (file stem or its prefix)
$PY -m experiment.pipeline.figures --list                  # figures and the tables they read
$PY -m experiment.pipeline report FIG                      # the same as drawing them all
```

Without `--runs-dir` or `$PSMC_EXP_RUNS`, `figures` and `report FIG`
read the committed tables in `experiment/runs` and write every figure over
its committed file in `runs/<EXP>/figures/`; `figures --out-dir DIR` writes
into DIR instead.

`FIG` has one job per figure, 16 in the committed runs. The five that read
E7's tables (`art1_agreement`, `art2_performance`, `fig7_1000_genomes`,
`figS5_1000g_populations` and `figS6_1000g_checks`) are declared only once
E7's data stage has written `<runs>/E7/inputs/MANIFEST.json`, or once every
table the figure reads is there (both hold in the committed runs). Each job
waits for its tables, so `start --runs-dir $RUNS E1 FIG` in a new runs
directory declares 11 figure jobs and draws `fig1_workflow` (a schematic,
which reads none), `fig2_simulated_accuracy` and `figS1_em_convergence`;
the other eight are blocked, their tables missing, and run at a `start`
with or after the experiments that write them. The article numbers its
figures in its own order: its Fig 1 is `art1_agreement` (from E1, E2 and
E7's tables), its Fig 2 `art2_performance` (from E3 and E7's), and its Fig
3 `fig7_1000_genomes`; the supplement's Figs S1, S2 and S3 are
`figS1_em_convergence`, `fig5_input_preparation` and
`fig6_bootstrap_calibration` (Supplementary Note 1, section 3). These six
are drawn in the house style of `style.py`; the article leaves the other
ten out.

## The figures

| figure | written to | reads | width x height |
| --- | --- | --- | --- |
| `art1_agreement` (the article's Fig 1) | `FIG/figures/` | `E1_curves`, `E1_table1`, `E1_replicates`, `E1_rs_vs_c`, `E2_curves`, `E2_bands`, `E2_rs_vs_c`, `E7_rs_vs_c` | 183 x 88 mm |
| `art2_performance` (the article's Fig 2) | `E3/figures/` | `E3_table2`, `E7_fits` | 183 x 55 mm |
| `fig1_workflow` | `FIG/figures/` | nothing (schematic) | 183 x 86 mm |
| `fig2_simulated_accuracy` | `E1/figures/` | `E1_curves`, `E1_truth`, `E1_table1`, `E1_replicates`, `E1_rs_vs_c`, `E1_mstep` | 183 x 150 mm |
| `fig3_real_genomes` | `E2/figures/` | `E2_curves`, `E2_bands`, `E2_bootstrap`, `E2_rs_vs_c`, `E2_fit_summary` | 183 x 112 mm |
| `fig4_performance` | `E3/figures/` | `E3_table2`, `E3_runs` | 183 x 70 mm |
| `fig5_input_preparation` (the supplement's Fig S2) | `E4/figures/` | `E4_routes`, `E4_curves` | 183 x 60 mm |
| `fig6_bootstrap_calibration` (the supplement's Fig S3) | `E5/figures/` | `E5_summary`, `E5_sampling`, `E5_bands`, `E5_sd_ratio`, `E5_coverage` | 183 x 56 mm |
| `figS1_em_convergence` (the supplement's Fig S1) | `E1/figures/` | `E1_traces` | 183 x 52 mm |
| `figS2_format_consistency` | `E6/figures/` | `E6_S2_curves`, `E6_S2_bins`, `E6_S2_fits` | 183 x 96 mm |
| `figS3_tmrca_decoding` | `E6/figures/` | `E6_S3_track`, `E6_S3_density`, `E6_S3_calibration`, `E6_S3_summary` | 183 x 118 mm |
| `figS4_thread_scaling` | `E3/figures/` | `E3_thread_scaling`, `E3_table2`, `E3_runs`, `E3_determinism` | 183 x 62 mm |
| `fig7_1000_genomes` (the article's Fig 3) | `E7/figures/` | `E7_pop_bands` | 183 x 99 mm |
| `figS5_1000g_populations` (not in the article; Supplementary Note 1, section 3) | `E7/figures/` | `E7_pop_bands`, `E7_samples` | 183 x 172 mm |
| `figS6_1000g_checks` (not in the article; Supplementary Note 1, section 3) | `E7/figures/` | `E7_inputs`, `E7_route_shift`, `E7_split`, `E7_fits`, `E7_curves`, `E7_giab`, `E7_throughput` | 183 x 110 mm |
| `figS_turtle_patterns` (not in the article; Supplementary Note 1, section 3) | `E2/figures/` | `E2_split_curves`, `E2_split_patterns` | 183 x 72 mm |

What each panel shows is in the module's docstring (`fig2.py`, ...), written
so that it can be turned into a caption. The notes below name each figure by
its file stem; "the article's Fig N" and "the supplement's Fig SN" are the
six the article includes, whose captions should also say what the notes
give. For the other ten the notes say what the figure shows.

- **Every psmc-rs fit** uses its default M-step, which runs to convergence
  as C psmc's does (`--mstep-iters 1000` is passed explicitly and is only
  a cap; `config.Inference`). Panel i of `fig2_simulated_accuracy`,
  `fig4_performance` and `figS4_thread_scaling` show a psmc-rs fit with
  capped M-steps only when `E1.default_mstep_iters` or
  `E3.default_mstep_iters` sets a cap, which production does not.
- **`fig2_simulated_accuracy`, `figS2_format_consistency` and the
  supplement's Figs S2 and S3** (`fig5_input_preparation`,
  `fig6_bootstrap_calibration`): times in years with mu 2.5e-8 and g 25; Ne
  in diploid individuals (both in thousands on the axes of the supplement's
  Figs S2 and S3). In `fig2_simulated_accuracy` and
  `figS2_format_consistency` grey shading marks times outside the model's
  informative window (1% to 99% quantiles of the true pairwise coalescence
  time), where no metric is computed; the supplement's Fig S2a and Fig S3
  draw that window only.
- **`fig2_simulated_accuracy`**: in a-d, Ne values of the fits outside the
  axis are clipped; a replicate's first or last free lambda, which the data
  barely constrain, can lie far above it. In expansion replicate 5 the first
  free lambda (intervals 0-3, before 4.8 ky, outside the window) ran away in
  both tools: psmc-rs's stopped at its upper bound (lambda 1e4, Ne 2.3e7;
  psmc-rs bounds every free lambda), and C psmc's, which is unbounded,
  reached about 1.1e6 (Ne 2.5e9). Both leave c at 4.8 ky, C psmc's line
  under psmc-rs's. The 2.0 named off scale in g, with C psmc higher
  (`E1_rs_vs_c.max_abs_log10_ne_rs_vs_c_any_interval`), is the gap between
  the two, set by psmc-rs's bound, not a disagreement on what the data say.
- **`fig2_simulated_accuracy` e-i, `fig3_real_genomes`,
  `figS2_format_consistency` e-h and the article's Fig 1**
  (`art1_agreement`'s Delta and e): two fits of the same pattern are
  compared interval by interval, not at fixed times: interval k of one with
  interval k of the other, drawn on the reference's intervals (C psmc's, the
  psmcfa fit's). Their boundaries differ in years only through theta and
  t_max, and a fixed time between two nearly equal boundaries would read a
  whole step. The RMSE weighs each interval by its log-time width inside the
  window.
- **The article's Fig 1** (`art1_agreement`): every curve and band of a-d
  is clipped to its genome's window: a simulated model's informative window
  (`E1_table1`), a real genome's comparison window (1% to 99% of
  coalescences under C psmc's fit, `E2_rs_vs_c`). Their grey spread of
  estimates is the range (pointwise minimum to maximum) of psmc-rs's fits
  to every replicate of a simulated model, and psmc-rs's bootstrap 95% band
  at the default blocks for a real genome. The fold on e's arrow ('7,000x
  smaller' in production) is `\EoneFoldMedian`: the median error of
  psmc-rs against the truth over the median Delta, on the simulated
  genomes, to one significant digit.
- **The article's Fig 2** (`art2_performance`): a and b are the medians of
  E3's clean repeats on the thread-scaling genome (30 chromosomes of 100 Mb
  in production), each run alone on its machine. psmc-rs fits each sequence
  on one thread, in parallel across sequences, so the speed-up on many
  threads needs as many sequences of similar length (see
  `fig4_performance`). c's times are one-thread fits of the 1000 Genomes
  cohort, both programs' in one job on one node shared with other fits
  (`E7_fits.wall_seconds`). Their ratio, of C psmc's median over the default
  genomes (52 in production) to psmc-rs's over every genome (2,504),
  compares the programs under the same conditions; their absolute values are
  not comparable with a's.
- **The article's Fig 3** (`fig7_1000_genomes`): medians over genomes of
  their psmcfa fits, every curve over the cohort's comparison window. A
  group whose own window starts later (in production East Asia, CHB, CHS
  and BEB) starts inside PSMC's first interval, a single parameter one
  genome constrains poorly, so the populations' spread there is not
  demography. A psmcfa takes every called bin as full, which makes theta,
  and every size and time, a little low (`fig7.py` gives the shift of the
  VCF route).
- **`fig3_real_genomes`**: each species with its own mu and g, in the panel
  with their source in a few words (in full:
  `config.RealGenome.scaling_source`, repeated in
  `E2_fit_summary.scaling_source`). Turtle curves are scaled with mu = 4.6 x
  10^-9 per bp per generation, the mean germline rate of nine *Chrysemys
  picta* trios (Bergeron et al. 2023; no pleurodire rate has been measured),
  and g = 15 years for all three species (close to Sozzoni et al. 2025's
  15.9 for *P. castaneus* and 12.4 for *E. subglobosa*; *H. tectifera* has
  none published); the human with mu = 1.25 x 10^-8 (Scally & Durbin 2012)
  and g = 29 years (Fenner 2005; as in Bergström et al. 2020 and Cousins et
  al. 2025). Sensitivity: Ne and time scale as 1/mu, and time also scales
  with g. Plausible turtle values (mu 3.0-7.9 x 10^-9 per generation, from
  the range of the *C. picta* trios up to the crocodilian rate of Green et
  al. 2014; g 12-20 years) span a factor of about 2.6 on both axes, and a
  further factor of 0.8-1.33 on the time axis. The human genome is NA12878
  (CEU; GIAB's HG001), chr1-22 of the 1000 Genomes 30x NYGC joint call with
  the GRCh38 strict mask and every non-PASS position missing: E7's psmcfa of
  that sample, byte for byte. E7 fits the same psmcfa with the same
  settings, so its NA12878 curves are this figure's, and E2 bootstraps it
  with E7's seed, replicates, trunks and iterations
  (`config.RealGenome.bootstrap_shared_with`), so its band is this figure's
  too. The bootstrap resamples blocks: the sequences split at 50 Mb
  (splitfa's trunks; a remainder under 1.5 trunks stays in the last block),
  so a sequence shorter than 50 Mb is one whole block. Whole scaffolds
  shorter than 50 Mb hold 97% of *P. castaneus*'s bins, 50% of *E.
  subglobosa*'s and 23% of *H. tectifera*'s, and only chr21 (1.6%) of
  NA12878's (`E2_bootstrap.bins_below_trunk_fraction`; NA12878's 22
  chromosomes form 59 blocks). The legend gives no block size, since few
  blocks are 50 Mb long; each panel gives its blocks and sequences. A genome
  whose default blocks are fewer than 20 is bootstrapped again on 5 Mb
  blocks (dark dotted outline; the legend then names both sizes), and its
  coarse 50 Mb band does not set the Ne axis and is clipped where it reaches
  beyond it (the panel says so); no genome of the production run has so few,
  so this figure has neither. E5 calibrates equal 50 Mb and 5 Mb blocks of
  50 Mb chromosomes, the scheme closest to NA12878's; the turtles' bands are
  not calibrated.
- **`fig3_real_genomes` and `figS_turtle_patterns`**: the turtle curves use
  the pattern `4+25*2+4+6`, under which Hilgers et al. 2025 (Curr Biol
  35:927) found false recent peaks of Ne in all their turtles (and across
  many PSMC studies): the first free lambda spans the first four intervals.
  `figS_turtle_patterns` fits the three turtles again with that window
  split (`2+2+25*2+4+6`, `1+1+1+1+25*2+4+6`; psmc-rs only, every other
  setting the same) and gives, per pattern, the largest Ne of the first 8
  intervals relative to the plateau of the next 8, marked "peak" where it
  is one (`E2_split_patterns`): whether `fig3_real_genomes`'s recent peaks
  persist when that window is split.
- **`fig4_performance`**: medians of E3's clean repeats (each repeat is a
  dot under its marker); sizes a tool was not run at are named in panel a
  with the reason E3 recorded. psmc-rs runs the E-step of each sequence on
  one thread, in parallel across sequences, so on t threads a genome of n
  equal chromosomes takes about ceil(n / t) times one chromosome's time:
  10.4 to 11.5 s for every psmc-rs 0.2 setup (Supplementary Note 1, section
  10). The genomes are 1, 3, 10 and 30 chromosomes of 100 Mb (the second
  line of the size ticks). Threads beyond the chromosome count add nothing:
  the 30-thread lines are flat, and the 30-thread speed-up at 3,000 Mb
  (384x) holds because its 30 equal chromosomes give each thread one. On a
  real genome the longest sequence sets the floor (`figS4_thread_scaling`).
  Panel b's peak memory is the larger of the tool's `ru_maxrss` and its
  sampled peak; for psmc-rs 0.2 on 100 and 300 Mb, and on 1000 Mb on one
  thread, it is `ru_maxrss`'s 26.0 MB in every repeat, above the sampled 7.0
  to 19.6 MB, so those points are an upper bound (Supplementary Note 1,
  section 10).
- **The supplement's Fig S2** (`fig5_input_preparation`): binning with
  partly callable bins biases PSMC, and psmc-rs's per-bin model removes most
  of the bias. With a dispersed mask almost every bin stays called with
  about (1 - f) of its bases; taking them as full makes theta, and every
  time and size, low by about 1.2 times the missing share of the called bins
  (E4: theta 12% low with 10% missing, 35% with 30%). The figure draws bins
  taken as full as the classic consensus FASTA -> fq2psmcfa route (C psmc's
  input and model, orange): its bins equal the VCF + mask bins read with
  `--bin-callable ignore` and its fit equals theirs (both asserted by E4's
  report; `fig5.full_route` checks theta), so `mask_ignore` is drawn only
  where the tables have no fq2psmcfa rows. Weighting each bin by its
  callable bases (`auto`, psmc-rs's default, blue) comes close to the
  complete-data fit (the wide grey underlay in a; E4: theta 0.7% low on
  average with 10% missing, 1.9% with 30%). Clustered missing data empty
  whole bins and change little (b's left facet). multihetsep (the uncalled
  sites of an interval are placed at its start, and every called bin is
  taken as full: `auto` fits exact counts only) and a VCF without its mask
  are not drawn; the caption gives their numbers (`E4_routes`). The
  complete-data fit is the control's VCF + mask fit, identical under both
  models (asserted).
- **The supplement's Fig S3** (`fig6_bootstrap_calibration`): a's band and
  the independent fits' range run off the top of the axis in the first
  interval (the Ne axis spans half the truth's smallest size to five times
  its largest). b's grey band is where a bootstrap with the true SD would
  fall, given the true SD's own chi-square 95% interval. In c a dot is one
  genome's share of the window's times at which its band holds the fits'
  mean or the truth, and a bar the mean over the genomes (`E5_summary`); the
  gap between the two groups is PSMC's bias, not the bootstrap's.
- **The supplement's Fig S1** (`figS1_em_convergence`): iteration i counts
  completed EM iterations (C psmc's LK of round i + 1, psmc-rs's
  `loglike[i]`); both programs are drawn up to the last iteration both give,
  as C psmc prints no log-likelihood after its last. Iteration 0 (the same
  start in both), the sign of the difference, and theta and rho are left to
  the caption. The arrows' ratios are floored to two significant figures, so
  each is a bound.

## Style (`style.py`)

The house style of the six article and supplement figures is set out in
full in `style.py`'s docstring; in short:

- **Type**: Arial (else Helvetica, else DejaVu Sans) in three sizes: 8 pt
  bold lowercase panel letters; 7 pt axis labels, panel titles, legend text
  and group headers; 6 pt tick labels, numbers and names printed on the data
  (in ink) and context notes (in a dark grey). Superscripts and subscripts
  at 0.84 of their text (`style.MATH_SHRINK`; mathtext's own 0.7 would set
  a 6 pt tick label's exponent at 4.2 pt), so no text is below 5 pt (the
  tests check every figure's SVG with `style.small_text`). Sizes and times
  on log axes are plain numbers in thousands.
- **Neutral greys by role** (ink `#1a1a1a`, spines `#404040`, notes
  `#595959`, reference lines `#8c8c8c`, context curves `#c4c4c4`, the one
  neutral band `#e8e8e8`, hairlines `#e6e6e6`): colour belongs to the data.
- **Line weights**: 0.35 pt hairlines, 0.5 pt axes, reference and context
  lines, 0.75 pt data lines (the truth included), 1.0 pt for at most one
  emphasised summary per panel, and a 2.4 pt pale underlay where two curves
  coincide (the underlay rule: C psmc's wide pale-orange line beneath
  psmc-rs's thin blue one; never a dashed line over a solid one).
- **Bands** are opaque tints, never alpha (psmc-rs's `#dde9f8`); the
  pipeline-only figures keep alpha 0.18 (`style.BAND_ALPHA`).
- **Layout**: no top or right spines, no dual axes, no grid on N_e(t)
  panels; letters and titles share one baseline 1.5 mm above the axes; a
  key, where the data cannot be named in place, is one line at the top.
  Widths are the journal's: 89, 120 and 183 mm, printed at 100%.

Colours follow the entity in every figure (a reference palette for light
mode; pairwise contrast on white and separation under colour-vision
deficiency were checked with an external palette validator, not in this
repository):

| entity | colour | check |
| --- | --- | --- |
| psmc-rs | `#2a78d6` blue; its band `#dde9f8` | pass, with C psmc's orange |
| C psmc | `#eb6834` orange; its underlay `#f6bba4` | |
| truth | `#1a1a1a` (ink), dashed | |
| a series named in place (no legend) | its text colour: C psmc `#c4501f`, psmc-rs `#1c5cab`, a superpopulation's `style.SUPERPOP_TEXT` | at least 4.3:1 on white |
| routes of the supplement's Fig S2 (`fig5_input_preparation`) | weighted bins (`mask_auto`) blue; bins taken as full through fq2psmcfa orange, from the VCF (`mask_ignore`) `#104281` (blue ramp 650); multihetsep `#b07800`, a VCF without its mask `#c94f80` (not drawn) | pass (ordered: blue, then `#104281`) |
| block sizes (the supplement's Fig S3, `fig6_bootstrap_calibration`) | the most blocks `#2a78d6`, the fewest `#86b6ef` (blue ramp) | |
| superpopulations (the article's Fig 3, `fig7_1000_genomes`) | AFR `#eda100`, AMR `#12966b`, EAS `#4a3aa7`, EUR `#e87ba4`, SAS `#a0522d` | pass (`fig7.py`'s docstring gives the separations) |
| a psmc-rs variant in the pipeline-only figures (VCF input, 5 Mb trunks, the split-first-window patterns) | `#104281` (blue ramp 650), dotted or dashed | |
| counts (`figS3_tmrca_decoding`, pipeline-only) | sequential blue ramp, steps 100-700 | |

## Tables (`schemas.py`)

The figures read only these tables, which the experiments' `report` jobs
write (and `config.py`: the demographic models, whose true Ne(t) is drawn
with its exact change times, and the figure settings in `config.FIGURES`).
`data.read` checks that the columns a figure needs are present and names the
missing ones; other columns may be there. The unit tests run every
experiment at the tiny profile and draw every figure from the tables it
writes, so a schema cannot drift from its table unnoticed. Curve tables have
one row per PSMC interval: `ne` holds from `years` to the next row's
`years` (the first is 0, the last interval runs to infinity), as
`results.Fit.curve` gives it. `tool` is `c` or `rs`, and in E3 `c`,
`rs02` or `rs02-default`.

| table | one row per | columns the figures need |
| --- | --- | --- |
| `E1/tables/E1_curves.csv` | model, replicate, tool and interval | `model`, `replicate`, `tool`, `interval`, `years`, `ne` |
| `E1/tables/E1_truth.csv` | model and time of a dense log grid | `model`, `years`, `ne` |
| `E1/tables/E1_table1.csv` | model | `model`, `window_lo_years`, `window_hi_years` |
| `E1/tables/E1_replicates.csv` | model, replicate and tool | `model`, `replicate`, `tool`, `rmse_log10_ne_vs_truth` |
| `E1/tables/E1_rs_vs_c.csv` | model and replicate | `model`, `replicate`, `rmse_log10_ne_rs_vs_c_per_interval`, `max_abs_log10_ne_rs_vs_c_per_interval`, `max_abs_log10_ne_rs_vs_c_any_interval` |
| `E1/tables/E1_mstep.csv` (no rows without a capped fit) | model and replicate | `model`, `replicate`, `mstep_iters_default`, `rmse_log10_ne_default_vs_c_per_interval` |
| `E1/tables/E1_traces.csv` | model, replicate, tool and EM iteration | `model`, `replicate`, `tool`, `iteration`, `loglike`, `theta`, `rho` |
| `E2/tables/E2_curves.csv` | genome, tool and interval | `sample_id`, `tool`, `interval`, `years`, `ne` |
| `E2/tables/E2_bands.csv` | genome, bootstrap trunk size and time of a dense log grid | `sample_id`, `trunk_bins`, `years`, `ne_band_lo`, `ne_band_hi` |
| `E2/tables/E2_bootstrap.csv` | genome and bootstrap trunk size | `sample_id`, `default_trunks`, `trunk_bins`, `trunk_mb`, `trunks`, `coarse_band`, `replicates` |
| `E2/tables/E2_rs_vs_c.csv` | genome | `sample_id`, `window_lo_years`, `window_hi_years`, `rmse_log10_ne_rs_vs_c_per_interval`, `max_abs_log10_ne_rs_vs_c_per_interval` |
| `E2/tables/E2_fit_summary.csv` | genome and tool | `sample_id`, `tool`, `mu`, `gen_years`, `scaling_cite`, `sequences` |
| `E2/tables/E2_split_patterns.csv` | turtle and pattern (`default`: `fit-rs`'s fit) | `sample_id`, `pattern_name`, `pattern`, `mu`, `gen_years`, `window_lo_years`, `window_hi_years`, `recent_end_years`, `first_vs_plateau`, `recent_max_vs_plateau`, `recent_peak` |
| `E2/tables/E2_split_curves.csv` | turtle, pattern and interval | `sample_id`, `pattern_name`, `interval`, `years`, `ne` |
| `E3/tables/E3_table2.csv` | tool, thread count and genome size (Table 2) | `tool`, `threads`, `size_mb`, `chromosomes`, `status`, `note`, `wall_median_s`, `peak_rss_median_mb`, `speedup_vs_c` |
| `E3/tables/E3_runs.csv` | timed run | `tool`, `threads`, `size_mb`, `repeat`, `wall_seconds`, `peak_rss_mb`, `clean` |
| `E3/tables/E3_thread_scaling.csv` | thread count (psmc-rs 0.2 on the thread-scaling genome) | `size_mb`, `threads`, `wall_median_s`, `efficiency`, `peak_rss_median_mb` |
| `E3/tables/E3_determinism.csv` | run, against its tool's reference at the same size | `tool`, `size_mb`, `threads`, `sha256`, `max_rel_diff_params` |
| `E4/tables/E4_routes.csv` | mask, route and replicate | `mask`, `route`, `replicate`, `theta_rel_complete`, `rmse_log10_ne_vs_truth`, `rmse_log10_ne_vs_complete` |
| `E4/tables/E4_curves.csv` | mask, route, replicate and interval | `mask`, `route`, `replicate`, `interval`, `years`, `ne` |
| `E5/tables/E5_summary.csv` | trunk size and window | `trunk_bins`, `trunk_mb`, `trunks`, `window`, `sd_ratio_median`, `coverage_mean`, `coverage_mean_ci_lo`, `coverage_mean_ci_hi`, `coverage_of_fit_mean_mean`, `coverage_projected_mean`, `coverage_projected_mean_ci_lo`, `coverage_projected_mean_ci_hi` |
| `E5/tables/E5_sampling.csv` | time of the informative window | `point`, `years`, `true_ne`, `fits`, `mean_log10_ne`, `lo_ne`, `hi_ne` |
| `E5/tables/E5_bands.csv` | bootstrapped genome, trunk size and time | `dataset`, `trunk_bins`, `point`, `years`, `fit_ne`, `boot_lo_ne`, `boot_hi_ne`, `covers_truth`, `projected_ne` |
| `E5/tables/E5_sd_ratio.csv` | trunk size and time | `trunk_bins`, `point`, `years`, `sd_true_log10`, `sd_true_ci_lo`, `sd_true_ci_hi`, `sd_boot_log10_median`, `ratio_median`, `ratio_q25`, `ratio_q75` |
| `E5/tables/E5_coverage.csv` | trunk size and time | `trunk_bins`, `point`, `years`, `datasets`, `coverage`, `coverage_ci_lo`, `coverage_ci_hi`, `coverage_fit_mean`, `coverage_projected`, `coverage_projected_ci_lo`, `coverage_projected_ci_hi` |
| `E6/tables/E6_S2_curves.csv` | model, format and interval (format truth: a dense grid of the true Ne) | `model`, `format`, `interval`, `years`, `ne` |
| `E6/tables/E6_S2_bins.csv` | model, format and sequence (all: the genome) | `model`, `format`, `sequence`, `bins_differing`, `bins_missing_at_end` |
| `E6/tables/E6_S2_fits.csv` | model and format | `model`, `format`, `window_lo_years`, `window_hi_years` |
| `E6/tables/E6_S3_track.csv` | 100 bp bin of the example window | `sequence`, `start_bp`, `obs`, `true_years`, `posterior_mean_years`, `q025_years`, `q975_years` |
| `E6/tables/E6_S3_density.csv` | non-empty cell of a log10 grid | `log10_true_lo`, `log10_true_hi`, `log10_mean_lo`, `log10_mean_hi`, `n_bins` |
| `E6/tables/E6_S3_calibration.csv` | class (decile) of the true TMRCA or of the posterior mean | `by`, `class`, `median_true_years`, `coverage95_states`, `coverage95_point` |
| `E6/tables/E6_S3_summary.csv` | set of bins (all, heterozygous, homozygous) | `bins`, `n_bins`, `pearson_log10_mean`, `coverage95_states`, `coverage95_point` |
| `E7/tables/E7_samples.csv` | sample of the cohort | `sample_id`, `population`, `superpopulation`, `admixed`, `raw_depth`, `default`, `bootstrap` |
| `E7/tables/E7_fits.csv` | sample and route | `sample_id`, `route`, `n0`, `threads`, `wall_seconds` |
| `E7/tables/E7_curves.csv` | sample, route and interval | `sample_id`, `route`, `interval`, `years`, `ne` |
| `E7/tables/E7_pop_bands.csv` | group, route and time on the display grid | `group_kind`, `group`, `superpopulation`, `admixed`, `route`, `n`, `years`, `ne_median`, `ne_lo`, `ne_hi`, `window_lo_years`, `window_hi_years` |
| `E7/tables/E7_rs_vs_c.csv` | sample of the default set fitted by both tools | `sample_id`, `population`, `superpopulation`, `admixed`, `window_lo_years`, `window_hi_years`, `rmse_log10_ne_rs_vs_c_per_interval`, `max_abs_log10_ne_rs_vs_c_per_interval` |
| `E7/tables/E7_inputs.csv` | sample of the cohort | `sample_id`, `population`, `superpopulation`, `admixed`, `default`, `raw_depth`, `callable_bases`, `het_sites`, `het_per_callable_kb`, `partly_callable_bin_fraction` |
| `E7/tables/E7_route_shift.csv` | sample of the default set | `sample_id`, `population`, `superpopulation`, `admixed`, `theta_rel_vcf_vs_psmcfa`, `mean_log10_ne_vcf_vs_psmcfa`, `partly_callable_bin_fraction`, `missing_share_of_called_bins` |
| `E7/tables/E7_split.csv` | sample of the split-first-window check | `sample_id`, `population`, `superpopulation`, `admixed`, `log10_first_vs_next_default`, `log10_first_vs_next_split`, `log10_first_vs_next_split_lo`, `log10_first_vs_next_split_hi`, `max_abs_log10_ne_split_vs_default_per_interval`, `ne_next_split` |
| `E7/tables/E7_giab.csv` | sample compared with GIAB | `sample_id`, `precision`, `recall`, `truth_het_snvs`, `eval_bases` |
| `E7/tables/E7_throughput.csv` | route and tool | `route`, `tool`, `jobs`, `threads`, `wall_median_s`, `core_hours`, `peak_rss_median_mb` |

## Production

The figures run last, as `FIG` jobs after each experiment's `report` job,
in the production run of [the pipeline's README](../README.md#production),
after E3 alone:

```bash
$PY -m experiment.pipeline start E1 E2 E4 E5 E6 FIG --cores 50 --mem-gb 64   # once E3's runner has exited
$PY -m experiment.pipeline start E7 FIG --cores 50 --mem-gb 64               # after E7's data stage
```

On fewer than 50 cores E2's bootstraps never start, nor its report, nor the
figures that read its tables. One job per figure (16), of a few seconds
each (1 core, 0.1-0.4 GB peak; about half a CPU minute in all: 16 figures
in 8 s on a laptop, 35 CPU seconds), none exclusive; `plan` counts 20 s
and 1 GB each (`config.FIGURES.job_seconds`, `job_gb`).
`fig4_performance` and `figS4_thread_scaling` read only E3's tables and
run at the first `start` after E3; the five that read E7's tables,
`art2_performance` (which also reads E3's) among them, get their jobs only
once E7's data stage has run, or once their tables are already there, and
run with `start E7 FIG`. Commit the `figures/` directories and
`FIG/provenance/` with the tables.
