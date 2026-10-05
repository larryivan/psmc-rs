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
| `fig2_simulated_accuracy` | `E1/figures/` | `E1_curves`, `E1_truth`, `E1_table1`, `E1_replicates`, `E1_rs_vs_c` | 183 x 150 mm |
| `fig3_real_genomes` | `E2/figures/` | `E2_curves`, `E2_summary` | 183 x 112 mm |
| `fig4_performance` | `E3/figures/` | `E3_runs`, `E3_threads` | 183 x 64 mm |
| `fig5_input_preparation` | `E4/figures/` | `E4_routes`, `E4_curves` | 183 x 120 mm |
| `fig6_bootstrap_calibration` | `E5/figures/` | `E5_calibration`, `E5_curves` | 183 x 108 mm |
| `figS1_em_convergence` | `E1/figures/` | `E1_traces` | 183 x 118 mm |
| `figS2_format_consistency` | `E6/figures/` | `E6_formats`, `E6_format_curves` | 183 x 92 mm |
| `figS3_tmrca_decoding` | `E6/figures/` | `E6_tmrca_track`, `E6_tmrca_hist2d`, `E6_tmrca_calibration`, `E6_tmrca_summary` | 183 x 112 mm |
| `figS4_thread_scaling` | `E3/figures/` | `E3_threads` | 183 x 60 mm |

What each panel shows is in the module's docstring (`fig2.py`, ...), written
so that it can be turned into the caption. Captions should also say:

- **Fig 2, 5, 6, S2**: times in years with mu 2.5e-8 and g 25; Ne in diploid
  individuals; grey shading marks times outside the model's informative
  window (1% to 99% quantiles of the true pairwise coalescence time), where
  no metric is computed. In Fig 2a-d, Ne values of the fits outside the axis
  are clipped; a replicate's first or last free lambda, which the data barely
  constrain, can lie far above it (e-h show by how much the tools differ).
- **Fig 2e-h and S2e-h**: the differences between two step functions with
  slightly different time boundaries (each tool has its own t_max) show
  narrow spikes at the boundaries; they are real and short.
- **Fig 3**: each species with its own mu and g (in the panel; the turtle
  values are unverified, see `config._TURTLE_SCALING_SOURCE`); the human
  sample is chromosome 1 only.
- **Fig 4**: medians of the repeats (each repeat is a dot under its marker);
  sizes a tool was not run at are named in the panel with the reason E3
  recorded.
- **Fig 6**: coverage over 10 datasets moves in steps of 10%.

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
| the psmc-rs variant that should coincide (fq2psmcfa route, VCF input, 5 Mb trunks) | `#104281` (blue ramp 650), dotted or dashed | `"#2a78d6,#104281" --ordinal`: pass |
| counts (Fig S3) | sequential blue ramp, steps 100-700 | |

Uncertainty bands are the entity's colour at alpha 0.18; the independent
fits' range in Fig 6 is neutral grey.

## Tables (`schemas.py`)

The figures read only these tables (and `config.py`: the demographic models,
whose true Ne(t) is drawn with its exact change times, and the figure
settings in `config.FIGURES`). `data.read` checks that the columns a figure
needs are present and names the missing ones; other columns may be there.
Curve tables have one row per PSMC interval: `ne` holds from `years` to the
next row's `years` (the first is 0, the last interval runs to infinity), as
`results.Fit.curve` gives it. `tool` is `c`, `rs` or `rs01`.

| table | one row per | columns the figures need |
| --- | --- | --- |
| `E1/tables/E1_curves.csv` | model, replicate, tool and interval | `model`, `replicate`, `tool`, `interval`, `years`, `ne` |
| `E1/tables/E1_truth.csv` | model and time of a dense log grid | `model`, `years`, `ne` (checked against `config.MODELS`) |
| `E1/tables/E1_table1.csv` | model | `model`, `window_lo_years`, `window_hi_years` |
| `E1/tables/E1_replicates.csv` | model, replicate and tool | `model`, `replicate`, `tool`, `rmse_log10_ne_vs_truth` |
| `E1/tables/E1_rs_vs_c.csv` | model and replicate | `model`, `replicate`, `rmse_log10_ne_rs_vs_c`, `max_abs_log10_ne_diff` |
| `E1/tables/E1_traces.csv` | model, replicate, tool and EM iteration | `model`, `replicate`, `tool`, `iteration`, `loglike`, `theta`, `rho` |
| `E2/tables/E2_curves.csv` | sample, tool, replicate and interval | `sample`, `tool`, `replicate` (0: the fit; 1..N: psmc-rs bootstrap), `interval`, `years`, `ne` |
| `E2/tables/E2_summary.csv` | sample | `sample`, `mu`, `gen_years`, `rmse_log10_ne_rs_vs_c`, `window_lo_years`, `window_hi_years`, `bootstrap_replicates` |
| `E3/tables/E3_runs.csv` | tool, threads, genome size and repeat | `tool` (c, rs01, rs02), `threads`, `genome_mb`, `repeat`, `status` (ok, skipped, failed), `wall_seconds`, `peak_rss_mb`, `note` (why skipped) |
| `E3/tables/E3_threads.csv` | thread count and repeat | `genome_mb`, `threads`, `repeat`, `wall_seconds`, `peak_rss_mb`, `json_sha256` |
| `E4/tables/E4_routes.csv` | replicate, mask and route | `replicate`, `mask` (none or a `MaskSpec` name), `structure`, `missing_fraction`, `route` (exact, fq2psmcfa, multihetsep, nomask), `theta_rel_bias` (against the exact route without missing data), `rmse_log10_ne_vs_truth`, `bins_differ_fraction` |
| `E4/tables/E4_curves.csv` | replicate, mask, route and interval | `replicate`, `mask`, `route`, `interval`, `years`, `ne` |
| `E5/tables/E5_calibration.csv` | trunk size and time of the window's grid | `trunk_bins`, `years`, `sd_true`, `sd_boot_median`, `sd_boot_q25`, `sd_boot_q75`, `sd_ratio_median`, `coverage`, `n_datasets` |
| `E5/tables/E5_curves.csv` | kind, dataset, trunk, replicate and interval | `kind` (fit, bootstrap), `dataset`, `trunk_bins` (0 for fit), `replicate` (0 for fit), `interval`, `years`, `ne` |
| `E6/tables/E6_formats.csv` | model and input format | `model`, `input` (psmcfa, vcf, multihetsep), `bins_differ`, `tail_bins_missing` |
| `E6/tables/E6_format_curves.csv` | model, input and interval | `model`, `input`, `interval`, `years`, `ne` |
| `E6/tables/E6_tmrca_track.csv` | bin of the example window | `start_bp`, `true_tmrca_years`, `post_mean_years`, `post_map_years`, `q025_years`, `q975_years` |
| `E6/tables/E6_tmrca_hist2d.csv` | estimator and cell of a log-log grid | `estimator` (mean, map), `true_lo_years`, `true_hi_years`, `est_lo_years`, `est_hi_years`, `count` |
| `E6/tables/E6_tmrca_calibration.csv` | class of true TMRCA | `true_lo_years`, `true_hi_years`, `bins`, `coverage` |
| `E6/tables/E6_tmrca_summary.csv` | estimator | `estimator`, `pearson_log10`, `coverage`, `bins` |

`fake.py` writes small stand-in tables in these schemas (the truth with
seeded noise, sized by a profile) for developing and testing the figures;
`fake.calibration` is the definition `E5_calibration.csv` is expected to
follow. They go to a scratch directory, never to `experiment/runs`:

```bash
$PY -m experiment.pipeline.figures.fake --runs-dir /tmp/fig-dev --profile production E2 E3 E4 E5 E6
$PY -m experiment.pipeline.figures --runs-dir /tmp/fig-dev
```

## Production

The figures run last, as `FIG` jobs after each experiment's `report` job:

```bash
experiment/.venv/bin/python -m experiment.pipeline start E1 E2 E3 E4 E5 E6 FIG --keep-awake
```

Ten jobs of a few seconds each (1 core, 0.1-0.4 GB peak; a few CPU
minutes in all), none exclusive; `plan` counts 20 s and 1 GB each
(`config.FIGURES.job_seconds`, `job_gb`). A figure whose experiment has not run yet
stays blocked on its missing tables and runs at the next `start`. Commit the
`figures/` directories and `FIG/provenance/` with the tables.
