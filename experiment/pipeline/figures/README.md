# Figures

`experiment/pipeline/figures/` draws the manuscript's figures from the
experiments' tables. Each figure is one module with `draw(source)`; the
experiment `FIG` (`experiments/fig.py`) runs each as a runner job whose
inputs are the tables it reads, so a figure is redrawn exactly when a table,
the figure's code, its settings or the font change. Every figure is written
as `experiment/runs/<EXP>/figures/<name>.{pdf,svg,png}`: vector PDF and SVG
with text as text (TrueType, `fonttype 42`; SVG `svg.fonttype none`) and a
600-dpi PNG, without creation dates, so the same tables give the same bytes.

```bash
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline start E1 FIG          # figures as runner jobs, after their tables
$PY -m experiment.pipeline.figures               # or draw every figure whose tables exist
$PY -m experiment.pipeline.figures fig2 figS1    # some of them (file stem or its prefix)
$PY -m experiment.pipeline.figures --list        # figures and the tables they read
$PY -m experiment.pipeline report FIG            # the same as drawing them all
```

A figure job is declared once every experiment it reads is implemented
(registered in `experiments.REGISTRY`); Fig 1, a schematic, needs none. With
E1 alone, `FIG` has three jobs (Fig 1, Fig 2, Fig S1); Figs 3-6 and S2-S4
appear as E2-E6 are added.

## The figures

| figure | written to | reads | width x height |
| --- | --- | --- | --- |
| `fig1_workflow` | `FIG/figures/` | nothing (schematic) | 183 x 82 mm |
| `fig2_simulated_accuracy` | `E1/figures/` | `E1_curves`, `E1_truth`, `E1_table1`, `E1_replicates`, `E1_rs_vs_c`, `E1_mstep` | 183 x 150 mm |
| `fig3_real_genomes` | `E2/figures/` | `E2_curves`, `E2_bands`, `E2_bootstrap`, `E2_rs_vs_c`, `E2_fit_summary` | 183 x 112 mm |
| `fig4_performance` | `E3/figures/` | `E3_table2`, `E3_runs` | 183 x 70 mm |
| `fig5_input_preparation` | `E4/figures/` | `E4_routes`, `E4_curves`, `E4_bias` | 183 x 122 mm |
| `fig6_bootstrap_calibration` | `E5/figures/` | `E5_summary`, `E5_sampling`, `E5_bands`, `E5_sd_ratio`, `E5_coverage` | 183 x 132 mm |
| `figS1_em_convergence` | `E1/figures/` | `E1_traces` | 183 x 118 mm |
| `figS2_format_consistency` | `E6/figures/` | `E6_S2_curves`, `E6_S2_bins`, `E6_S2_fits` | 183 x 96 mm |
| `figS3_tmrca_decoding` | `E6/figures/` | `E6_S3_track`, `E6_S3_density`, `E6_S3_calibration`, `E6_S3_summary` | 183 x 118 mm |
| `figS4_thread_scaling` | `E3/figures/` | `E3_thread_scaling`, `E3_table2`, `E3_runs`, `E3_determinism` | 183 x 62 mm |

What each panel shows is in the module's docstring (`fig2.py`, ...), written
so that it can be turned into the caption. Captions should also say:

- **Every psmc-rs fit** uses its default M-step, which runs to convergence
  as C psmc's does (`--mstep-iters 1000` is passed explicitly and is only
  a cap; `config.Inference`). Fig 2i, Fig 4 and Fig S4 show a psmc-rs fit
  with capped M-steps only when `E1.default_mstep_iters` or
  `E3.default_mstep_iters` sets a cap, which production does not.
- **Fig 2, 5, 6, S2**: times in years with mu 2.5e-8 and g 25; Ne in diploid
  individuals; grey shading marks times outside the model's informative
  window (1% to 99% quantiles of the true pairwise coalescence time), where
  no metric is computed. In Fig 2a-d, Ne values of the fits outside the axis
  are clipped; a replicate's first or last free lambda, which the data barely
  constrain, can lie far above it. In expansion replicate 5, C psmc's first
  lambda ran away to about 1e6 (psmc-rs bounds every free lambda at 1e4):
  2.0 in log10 Ne before the window, which leaves Fig 2c and is named, off
  scale, in Fig 2g (`E1_rs_vs_c.max_abs_log10_ne_rs_vs_c_any_interval`).
- **Fig 2e-i, 3 and S2e-h**: two fits of the same pattern are compared
  interval by interval, not at fixed times: interval k of one with interval k
  of the other, drawn on the reference's intervals (C psmc's, the psmcfa
  fit's). Their boundaries differ in years only through theta and t_max, and
  a fixed time between two nearly equal boundaries would read a whole step.
  The RMSE weighs each interval by its log-time width inside the window.
- **Fig 3**: each species with its own mu and g (in the panel; the turtle
  values are unverified, see `config._TURTLE_SCALING_SOURCE`); the human
  sample is chromosome 1 only. The bootstrap resamples blocks: the
  sequences split at 50 Mb (splitfa's trunks), so most of the turtles'
  blocks are whole scaffolds. A genome whose default blocks are fewer than
  20 (Papuan chr1: 5 of 50 Mb) is bootstrapped again on 5 Mb blocks (dark
  dotted outline); its coarse 50 Mb band is clipped. E5 calibrates
  chromosome-scale blocks only.
- **Fig 4**: medians of E3's clean repeats (each repeat is a dot under its
  marker); sizes a tool was not run at are named in panel a with the reason
  E3 recorded.
- **Fig 5**: binning with partly callable bins biases PSMC, and psmc-rs's
  per-bin model removes most of the bias. With a dispersed mask almost
  every bin stays called with about (1 - f) of its bases; taking them as
  full (`--bin-callable ignore`, dark blue dashed) makes theta, and every
  time and size, low by about the missing fraction f or more, as the
  classic consensus FASTA -> fq2psmcfa route does: its bins equal the
  VCF + mask bins and its fit equals the dark blue one, both asserted, so
  it is not drawn. Weighing each bin by its callable bases (`auto`,
  psmc-rs's default, blue) comes close to the complete-data fit (grey in
  d-f; theta 2-4% low on average at 30% missing in
  `experiment/validation/`). Clustered missing data empty whole bins and
  change little; multihetsep (the uncalled sites of an interval are placed
  at its start, and every called bin is taken as full: `auto` fits exact
  counts only) and a VCF without its mask are shown as they are. The
  complete-data fit is the control's VCF + mask fit, identical under both
  models (asserted).
- **Fig 6**: coverage over 10 genomes moves in steps of 10% (shaded:
  pointwise Clopper-Pearson 95% interval); the window-averaged coverage in
  the legend has a 95% interval from resampling the genomes.

## Style (`style.py`)

Arial (else Helvetica, else DejaVu Sans), 7 pt text, 6 pt tick labels, 8 pt
bold lowercase panel letters; data lines 0.75-1.25 pt (thin replicate lines
0.5 pt; C psmc's medians 2 pt underneath psmc-rs's 0.8 pt where the two
coincide); no top or right spines, no titles, no dual axes; hairline grids
on log axes. Widths are the journal's: 89, 120 and 183 mm.

Colours follow the entity in every figure (dataviz reference palette, light
mode, checked with `dataviz/scripts/validate_palette.js`):

| entity | colour | check |
| --- | --- | --- |
| psmc-rs 0.2 | `#2a78d6` blue | `"#2a78d6,#eb6834,#1baf7a" --pairs all`: all pass (CVD 9.2, normal vision 24.0) |
| C psmc | `#eb6834` orange | |
| psmc-rs 0.1 (performance only) | `#1baf7a` aqua | below 3:1 on white: markers and a legend, Table 2 as the table view |
| truth | `#0b0b0b`, dashed | |
| multihetsep route / input (Fig 5, S2) | `#eda100` (slot 4) | `"#2a78d6,#eda100,#e87ba4" --pairs all`: pass (CVD 13.0, normal 19.6); both below 3:1, so they also carry their markers along the lines |
| VCF without a mask (Fig 5) | `#e87ba4` (slot 5) | |
| a psmc-rs variant of the default (every called bin full and the fq2psmcfa route in Fig 5, VCF input, 5 Mb trunks) | `#104281` (blue ramp 650), dotted or dashed | `"#2a78d6,#104281" --ordinal`: pass |
| counts (Fig S3) | sequential blue ramp, steps 100-700 | |

Uncertainty bands are the entity's colour at alpha 0.18; the independent
fits' range in Fig 6 is neutral grey.

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
`rs01`, `rs02` or `rs02-default`.

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
| `E2/tables/E2_fit_summary.csv` | genome and tool | `sample_id`, `tool`, `mu`, `gen_years`, `sequences` |
| `E3/tables/E3_table2.csv` | tool, thread count and genome size (Table 2) | `tool`, `threads`, `size_mb`, `status`, `note`, `wall_median_s`, `peak_rss_median_mb`, `speedup_vs_c` |
| `E3/tables/E3_runs.csv` | timed run | `tool`, `threads`, `size_mb`, `repeat`, `wall_seconds`, `peak_rss_mb`, `clean` |
| `E3/tables/E3_thread_scaling.csv` | thread count (psmc-rs 0.2 on the thread-scaling genome) | `size_mb`, `threads`, `wall_median_s`, `efficiency`, `peak_rss_median_mb` |
| `E3/tables/E3_determinism.csv` | run, against its tool's reference at the same size | `tool`, `size_mb`, `threads`, `sha256` |
| `E4/tables/E4_routes.csv` | mask, route and replicate | `mask`, `route`, `replicate`, `theta_rel_complete`, `rmse_log10_ne_vs_truth`, `rmse_log10_ne_vs_complete` |
| `E4/tables/E4_curves.csv` | mask, route, replicate and interval | `mask`, `route`, `replicate`, `interval`, `years`, `ne` |
| `E4/tables/E4_bias.csv` | mask, route and time of the informative window | `mask`, `route`, `years`, `replicates`, `log10_bias_mean` |
| `E5/tables/E5_summary.csv` | trunk size and window | `trunk_bins`, `trunk_mb`, `trunks`, `window`, `coverage_mean`, `coverage_mean_ci_lo`, `coverage_mean_ci_hi`, `coverage_projected_mean`, `coverage_projected_mean_ci_lo`, `coverage_projected_mean_ci_hi` |
| `E5/tables/E5_sampling.csv` | time of the informative window | `point`, `years`, `true_ne`, `fits`, `lo_ne`, `hi_ne` |
| `E5/tables/E5_bands.csv` | bootstrapped genome, trunk size and time | `dataset`, `trunk_bins`, `point`, `years`, `fit_ne`, `boot_lo_ne`, `boot_hi_ne`, `projected_ne` |
| `E5/tables/E5_sd_ratio.csv` | trunk size and time | `trunk_bins`, `point`, `years`, `sd_true_log10`, `sd_true_ci_lo`, `sd_true_ci_hi`, `sd_boot_log10_median`, `ratio_median`, `ratio_q25`, `ratio_q75` |
| `E5/tables/E5_coverage.csv` | trunk size and time | `trunk_bins`, `point`, `years`, `datasets`, `coverage`, `coverage_ci_lo`, `coverage_ci_hi`, `coverage_fit_mean`, `coverage_projected`, `coverage_projected_ci_lo`, `coverage_projected_ci_hi` |
| `E6/tables/E6_S2_curves.csv` | model, format and interval (format truth: a dense grid of the true Ne) | `model`, `format`, `interval`, `years`, `ne` |
| `E6/tables/E6_S2_bins.csv` | model, format and sequence (all: the genome) | `model`, `format`, `sequence`, `bins_differing`, `bins_missing_at_end` |
| `E6/tables/E6_S2_fits.csv` | model and format | `model`, `format`, `window_lo_years`, `window_hi_years` |
| `E6/tables/E6_S3_track.csv` | 100 bp bin of the example window | `sequence`, `start_bp`, `obs`, `true_years`, `posterior_mean_years`, `q025_years`, `q975_years` |
| `E6/tables/E6_S3_density.csv` | non-empty cell of a log10 grid | `log10_true_lo`, `log10_true_hi`, `log10_mean_lo`, `log10_mean_hi`, `n_bins` |
| `E6/tables/E6_S3_calibration.csv` | class (decile) of the true TMRCA or of the posterior mean | `by`, `class`, `median_true_years`, `coverage95_states`, `coverage95_point` |
| `E6/tables/E6_S3_summary.csv` | set of bins (all, heterozygous, homozygous) | `bins`, `n_bins`, `pearson_log10_mean`, `coverage95_states`, `coverage95_point` |

## Production

The figures run last, as `FIG` jobs after each experiment's `report` job,
in the production run of [the pipeline's README](../README.md#production):

```bash
$PY -m experiment.pipeline start E1 E2 E4 E5 E6 FIG
```

Ten jobs of a few seconds each (1 core, 0.1-0.4 GB peak; a few CPU
minutes in all), none exclusive; `plan` counts 20 s and 1 GB each
(`config.FIGURES.job_seconds`, `job_gb`). A figure whose experiment has not
run yet (Fig 4 and Fig S4 until E3 is done) stays blocked on its missing
tables and runs at the next `start`. Commit the
`figures/` directories and `FIG/provenance/` with the tables.
