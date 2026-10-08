"""The tables the figures read: file, columns and meaning.

These are tables the experiments' ``report`` jobs write to
``<runs>/<EXP>/tables/`` (``experiments/e1.py`` ... ``e6.py``); the schemas
list only the columns the figures read, and a figure checks that they are
present (:func:`.data.read`), naming the table and the missing columns
otherwise. Extra columns are allowed. The unit tests run every experiment
at the tiny profile and draw every figure from the tables it writes, so a
schema cannot drift from its table unnoticed.

Conventions, as in the experiments' tables: times in ``years`` (scaled with
the experiment's mu and g), sizes ``ne`` in diploid individuals, replicates
and datasets counted from 1, ``tool`` is ``c`` (C psmc), ``rs`` (psmc-rs
0.2) or, in E3, ``rs01`` (psmc-rs 0.1), ``rs02`` and ``rs02-default``. A
curve table has one row per PSMC interval: ``ne`` holds from ``years`` to
the next row's ``years`` (the first row's ``years`` is 0, the last interval
runs to infinity), as :meth:`..results.Fit.curve` gives it. Booleans are
``True``/``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

CURVE_COLUMNS = {
    "interval": "PSMC interval k (0-based)",
    "years": "start of interval k in years ago (0 for k = 0)",
    "ne": "Ne of interval k (diploid individuals)",
}


@dataclass(frozen=True)
class TableSchema:
    experiment: str
    file: str
    columns: Tuple[Tuple[str, str], ...]  # (name, meaning), in order
    rows: str  # what one row is

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(c for c, _ in self.columns)

    def path(self, runs_dir: Path) -> Path:
        return Path(runs_dir) / self.experiment / "tables" / self.file

    def describe(self) -> str:
        cols = "\n".join(f"    {c}: {m}" for c, m in self.columns)
        return f"{self.experiment}/tables/{self.file} (one row per {self.rows}):\n{cols}"


SCHEMAS: Dict[str, TableSchema] = {}


def _add(key: str, experiment: str, rows: str, columns: Dict[str, str]) -> None:
    SCHEMAS[key] = TableSchema(experiment, f"{key}.csv", tuple(columns.items()), rows)


# ---------------------------------------------------------------------------
# E1: simulated accuracy and equivalence (Fig 2, Fig S1)
# ---------------------------------------------------------------------------

_add("E1_curves", "E1", "model, replicate, tool and interval", {
    "model": "demographic model (config.MODELS)",
    "replicate": "replicate (1..)",
    "tool": "c or rs",
    **CURVE_COLUMNS,
})
_add("E1_truth", "E1", "model and time of a dense log grid", {
    "model": "demographic model",
    "years": "years ago",
    "ne": "true Ne at that time",
})
_add("E1_table1", "E1", "model", {
    "model": "demographic model",
    "window_lo_years": "start of the informative window (1% quantile of the true coalescence time)",
    "window_hi_years": "end of the informative window (99% quantile)",
})
_add("E1_replicates", "E1", "model, replicate and tool", {
    "model": "demographic model",
    "replicate": "replicate",
    "tool": "c or rs",
    "rmse_log10_ne_vs_truth": "RMSE of log10 Ne against the truth on the informative window",
})
_add("E1_rs_vs_c", "E1", "model and replicate", {
    "model": "demographic model",
    "replicate": "replicate",
    "rmse_log10_ne_rs_vs_c_per_interval": "RMSE of log10(Ne_rs,k / Ne_c,k) over C psmc's intervals k overlapping "
                                          "the informative window, each weighted by its log-time width inside it",
    "max_abs_log10_ne_rs_vs_c_per_interval": "largest |log10(Ne_rs,k / Ne_c,k)| of those intervals",
    "max_abs_log10_ne_rs_vs_c_any_interval": "largest |log10(Ne_rs,k / Ne_c,k)| of any interval, outside the window too",
})
_add("E1_mstep", "E1", "model and replicate", {
    "model": "demographic model",
    "replicate": "replicate",
    "mstep_iters_default": "the M-step cap of the capped psmc-rs fit (E1.default_mstep_iters)",
    "rmse_log10_ne_default_vs_c_per_interval": "RMSE of log10 Ne between psmc-rs with capped M-steps and C psmc, "
                                               "interval by interval (as rmse_log10_ne_rs_vs_c_per_interval)",
})
_add("E1_traces", "E1", "model, replicate, tool and EM iteration", {
    "model": "demographic model",
    "replicate": "replicate",
    "tool": "c or rs",
    "iteration": "completed EM iterations i (0 = the start)",
    "loglike": "log-likelihood after i iterations (empty for C at i = N_ITER: it prints none)",
    "theta": "theta per bin after i iterations",
    "rho": "rho per bin after i iterations",
})

# ---------------------------------------------------------------------------
# E2: real genomes (Fig 3)
# ---------------------------------------------------------------------------

_add("E2_curves", "E2", "genome, tool and interval", {
    "sample_id": "config.RealGenome.sample_id (HLemySub1, HLhydTec1, HLpelCas1, NA12878)",
    "tool": "c or rs",
    **CURVE_COLUMNS,
})
_add("E2_bands", "E2", "genome, bootstrap trunk size and time of a dense log grid", {
    "sample_id": "genome",
    "trunk_bins": "the bootstrap's trunk size (bins)",
    "years": "years ago",
    "ne_band_lo": "2.5% quantile of the bootstrap replicates' Ne",
    "ne_band_hi": "97.5% quantile",
})
_add("E2_bootstrap", "E2", "genome and bootstrap trunk size", {
    "sample_id": "genome",
    "default_trunks": "True for psmc-rs's default trunks, False for the second bootstrap on fine trunks",
    "trunk_bins": "trunk size (bins)",
    "trunk_mb": "trunk size (Mb)",
    "trunks": "trunks the genome is cut into",
    "coarse_band": "True when fewer than psmc-rs's 20 trunks",
    "replicates": "bootstrap replicates fitted",
})
_add("E2_rs_vs_c", "E2", "genome", {
    "sample_id": "genome",
    "window_lo_years": "start of the comparison window (1% of coalescences under C psmc's fit)",
    "window_hi_years": "end of the comparison window (99%)",
    "rmse_log10_ne_rs_vs_c_per_interval": "RMSE of log10(Ne_rs,k / Ne_c,k) over C psmc's intervals k overlapping "
                                          "the window, each weighted by its log-time width inside it",
    "max_abs_log10_ne_rs_vs_c_per_interval": "largest |log10(Ne_rs,k / Ne_c,k)| of those intervals",
})
_add("E2_fit_summary", "E2", "genome and tool", {
    "sample_id": "genome",
    "tool": "c or rs",
    "mu": "mutation rate the curves are scaled with",
    "gen_years": "generation time the curves are scaled with",
    "scaling_cite": "where mu (and g) come from, in a few words (config.RealGenome.scaling_cite)",
    "sequences": "sequences of the input (the bootstrap splits each at the block size)",
})
_add("E2_split_patterns", "E2", "turtle and pattern (default: fit-rs's fit)", {
    "sample_id": "genome (config.E2.split_genomes)",
    "pattern_name": "default, or a name of config.E2.split_patterns",
    "pattern": "the pattern fitted",
    "mu": "mutation rate the curves are scaled with",
    "gen_years": "generation time the curves are scaled with",
    "window_lo_years": "start of the comparison window (1% of coalescences under C psmc's default fit)",
    "window_hi_years": "end of the comparison window (99%)",
    "recent_end_years": "end of the recent intervals (config.E2.split_recent_intervals) in years",
    "first_vs_plateau": "Ne of the first interval / the plateau (median Ne of the intervals after the recent ones)",
    "recent_max_vs_plateau": "largest Ne of the recent intervals / the plateau",
    "recent_peak": "True when the largest recent Ne lies after the first interval and is at least "
                   "config.E2.split_peak_ratio times both the first interval's Ne and the plateau",
})
_add("E2_split_curves", "E2", "turtle, pattern and interval", {
    "sample_id": "genome",
    "pattern_name": "default, or a name of config.E2.split_patterns",
    **CURVE_COLUMNS,
})

# ---------------------------------------------------------------------------
# E3: performance (Fig 4, Fig S4)
# ---------------------------------------------------------------------------

_add("E3_table2", "E3", "tool, thread count and genome size (Table 2)", {
    "tool": "c, rs01, rs02 or rs02-default",
    "threads": "threads given to the tool (1 for C psmc)",
    "size_mb": "genome size (Mb)",
    "chromosomes": "chromosomes of the genome (100 Mb each in production)",
    "status": "measured, skipped (estimated memory above the limit) or incomplete",
    "note": "why a setup was skipped or is incomplete",
    "wall_median_s": "median wall time of the clean repeats",
    "peak_rss_median_mb": "median peak RSS of the process tree (MiB)",
    "speedup_vs_c": "C psmc's median wall time over this setup's, at the same size",
})
_add("E3_runs", "E3", "timed run", {
    "tool": "c, rs01, rs02 or rs02-default",
    "threads": "threads",
    "size_mb": "genome size (Mb)",
    "repeat": "repeat (1..)",
    "wall_seconds": "wall time",
    "peak_rss_mb": "peak RSS of the tool's process tree (MiB)",
    "clean": "True when the run was clean by the timing protocol: a quiet machine before it, little CPU used by "
             "other processes during it (other_method), no sleep (only clean runs enter the medians)",
})
_add("E3_thread_scaling", "E3", "thread count (psmc-rs 0.2 on the thread-scaling genome)", {
    "size_mb": "genome size (Mb)",
    "threads": "--threads",
    "wall_median_s": "median wall time",
    "efficiency": "speed-up over the fewest threads / threads",
    "peak_rss_median_mb": "median peak RSS (MiB)",
})
_add("E3_determinism", "E3", "run, against its tool's reference at the same size", {
    "tool": "c, rs01, rs02 or rs02-default",
    "size_mb": "genome size (Mb)",
    "threads": "threads",
    "sha256": "SHA-256 of the output",
})

# ---------------------------------------------------------------------------
# E4: input preparation (Fig 5)
# ---------------------------------------------------------------------------

_add("E4_routes", "E4", "mask, route and replicate", {
    "mask": "nomissing (the control) or a config.MaskSpec name",
    "route": "mask_auto, mask_ignore, fq2psmcfa, multihetsep or nomask",
    "replicate": "replicate",
    "theta_rel_complete": "theta / theta of the complete-data fit of the same genome (the control's mask_auto) - 1",
    "rmse_log10_ne_vs_truth": "RMSE of log10 Ne against the truth on the informative window",
    "rmse_log10_ne_vs_complete": "RMSE of log10 Ne against the complete-data fit of the same genome, same window, "
                                 "exact over log time (metrics.log_time_diff)",
})
_add("E4_curves", "E4", "mask, route, replicate and interval", {
    "mask": "mask",
    "route": "route",
    "replicate": "replicate",
    **CURVE_COLUMNS,
})
_add("E4_bias", "E4", "mask, route and time of the informative window", {
    "mask": "mask",
    "route": "route",
    "years": "years ago",
    "replicates": "replicates",
    "log10_bias_mean": "mean over replicates of log10(fitted Ne / true Ne)",
})

# ---------------------------------------------------------------------------
# E5: bootstrap calibration (Fig 6)
# ---------------------------------------------------------------------------

_add("E5_summary", "E5", "trunk size and window", {
    "trunk_bins": "--bootstrap-block-size (bins)",
    "trunk_mb": "trunk size (Mb)",
    "trunks": "trunks per genome",
    "window": "informative or fixed",
    "coverage_mean": "pointwise coverage averaged over the window",
    "coverage_mean_ci_lo": "its 2.5% bound from resampling the bootstrapped genomes",
    "coverage_mean_ci_hi": "its 97.5% bound",
    "coverage_projected_mean": "pointwise coverage of each genome's pattern-projected truth, averaged over the window",
    "coverage_projected_mean_ci_lo": "its 2.5% bound from resampling the bootstrapped genomes",
    "coverage_projected_mean_ci_hi": "its 97.5% bound",
})
_add("E5_sampling", "E5", "time of the informative window", {
    "point": "index of the time",
    "years": "years ago",
    "true_ne": "true Ne",
    "fits": "independent genomes fitted",
    "lo_ne": "2.5% quantile of the fits' Ne",
    "hi_ne": "97.5% quantile of the fits' Ne",
})
_add("E5_bands", "E5", "bootstrapped genome, trunk size and time", {
    "dataset": "genome (1..)",
    "trunk_bins": "trunk size (bins)",
    "point": "index of the time",
    "years": "years ago",
    "fit_ne": "the genome's fitted Ne",
    "boot_lo_ne": "2.5% quantile of its bootstrap replicates",
    "boot_hi_ne": "97.5% quantile",
    "projected_ne": "the true Ne projected onto the free-lambda spans of the genome's fit (metrics.projected_truth)",
})
_add("E5_sd_ratio", "E5", "trunk size and time", {
    "trunk_bins": "trunk size (bins)",
    "point": "index of the time",
    "years": "years ago",
    "sd_true_log10": "SD of log10 Ne over the independent fits (the true sampling SD)",
    "sd_true_ci_lo": "its chi-square 95% interval, lower bound",
    "sd_true_ci_hi": "upper bound",
    "sd_boot_log10_median": "median over the bootstrapped genomes of the bootstrap SD",
    "ratio_median": "median over the genomes of bootstrap SD / true SD",
    "ratio_q25": "its 25% quantile",
    "ratio_q75": "its 75% quantile",
})
_add("E5_coverage", "E5", "trunk size and time", {
    "trunk_bins": "trunk size (bins)",
    "point": "index of the time",
    "years": "years ago",
    "datasets": "bootstrapped genomes",
    "coverage": "share of them whose 95% band holds the true Ne",
    "coverage_ci_lo": "Clopper-Pearson 95% interval, lower bound",
    "coverage_ci_hi": "upper bound",
    "coverage_fit_mean": "share whose band holds the fits' geometric mean",
    "coverage_projected": "share whose band holds its own pattern-projected truth (metrics.projected_truth of its fit)",
    "coverage_projected_ci_lo": "Clopper-Pearson 95% interval, lower bound",
    "coverage_projected_ci_hi": "upper bound",
})

# ---------------------------------------------------------------------------
# E6: supplementary analyses (Fig S2, S3)
# ---------------------------------------------------------------------------

_add("E6_S2_curves", "E6", "model, format and interval (format truth: a dense grid of the true Ne)", {
    "model": "demographic model (E1 replicate config.E6.format_replicate)",
    "format": "psmcfa, vcf, multihetsep or truth",
    **CURVE_COLUMNS,
})
_add("E6_S2_bins", "E6", "model, format and sequence (all: the genome)", {
    "model": "demographic model",
    "format": "vcf or multihetsep",
    "sequence": "a sequence, or all",
    "bins_differing": "bins whose state differs from the psmcfa's",
    "bins_missing_at_end": "psmcfa bins after the format's last bin (the multihetsep tail)",
})
_add("E6_S2_fits", "E6", "model and format", {
    "model": "demographic model",
    "format": "psmcfa, vcf or multihetsep",
    "window_lo_years": "start of the model's informative window",
    "window_hi_years": "end of the window",
})
_add("E6_S3_track", "E6", "100 bp bin of the example window", {
    "sequence": "chromosome",
    "start_bp": "0-based start of the bin",
    "obs": "T, K or N",
    "true_years": "true pairwise TMRCA of the bin (length-weighted mean), years",
    "posterior_mean_years": "posterior mean TMRCA",
    "q025_years": "2.5% posterior quantile (its state's average time)",
    "q975_years": "97.5% posterior quantile",
})
_add("E6_S3_density", "E6", "non-empty cell of a log10 grid", {
    "log10_true_lo": "cell's lower edge, log10 true TMRCA (years)",
    "log10_true_hi": "upper edge",
    "log10_mean_lo": "cell's lower edge, log10 posterior mean",
    "log10_mean_hi": "upper edge",
    "n_bins": "bins in the cell",
})
_add("E6_S3_calibration", "E6", "class (decile) of the true TMRCA or of the posterior mean", {
    "by": "true or posterior_mean",
    "class": "decile (1..10)",
    "median_true_years": "median true TMRCA in the class",
    "coverage95_states": "share whose interval of states (2.5% to 97.5% quantile) holds the true TMRCA",
    "coverage95_point": "share whose quantiles' state average times hold it",
})
_add("E6_S3_summary", "E6", "set of bins (all, heterozygous, homozygous)", {
    "bins": "all, heterozygous (K) or homozygous (T)",
    "n_bins": "bins",
    "pearson_log10_mean": "Pearson correlation of log10 posterior mean and log10 true TMRCA",
    "coverage95_states": "coverage of the 95% interval of states",
    "coverage95_point": "coverage of the quantiles' average times",
})


# ---------------------------------------------------------------------------
# E7: 1000 Genomes (Fig 7, Figs S5 and S6)
# ---------------------------------------------------------------------------

_E7_WHO = {
    "sample_id": "1000 Genomes sample",
    "population": "population code (26)",
    "superpopulation": "AFR, AMR, EAS, EUR or SAS",
    "admixed": "the population is one of config.E7.admixed",
}
_add("E7_samples", "E7", "sample of the cohort", {
    "sample_id": "1000 Genomes sample",
    "population": "population code",
    "superpopulation": "superpopulation",
    "admixed": "admixed population",
    "raw_depth": "raw sequencing depth (the sequence index's bases / 3.1 Gb)",
    "default": "in the default set (the VCF fits, C psmc, the split check)",
    "bootstrap": "bootstrapped (Fig 7 g-i)",
})
_add("E7_pop_bands", "E7", "group, route and time on the display grid", {
    "group_kind": "population, superpopulation or all",
    "group": "the population or superpopulation code, or all",
    "superpopulation": "the group's superpopulation ('' for all)",
    "admixed": "an admixed population",
    "route": "rs-vcf (the default set's VCF fits) or rs-psmcfa (every genome's psmcfa fit)",
    "n": "genomes in the group",
    "years": "time (years ago)",
    "ne_median": "median Ne across the group's genomes",
    "ne_lo": "lower quantile (config.E7.spread) across genomes",
    "ne_hi": "upper quantile across genomes",
    "window_lo_years": "the group's median comparison window start",
    "window_hi_years": "its median end",
})
_add("E7_bands", "E7", "bootstrapped sample and time", {
    "sample_id": "sample",
    "years": "time (years ago)",
    "ne_c": "C psmc's Ne",
    "ne_rs": "psmc-rs's Ne (fit-rs-psmcfa)",
    "ne_band_lo": "bootstrap band lower edge",
    "ne_band_median": "bootstrap median",
    "ne_band_hi": "bootstrap band upper edge",
})
_add("E7_rs_vs_c", "E7", "sample of the default set fitted by both tools", {
    **_E7_WHO,
    "window_lo_years": "comparison window start (1% of coalescences under C psmc's fit)",
    "window_hi_years": "its end (99%)",
    "rmse_log10_ne_rs_vs_c_per_interval": "RMSE of log10 Ne between the tools, interval by interval in the window",
    "max_abs_log10_ne_rs_vs_c_per_interval": "largest per-interval |log10 Ne difference| in the window",
})
_add("E7_inputs", "E7", "sample of the cohort", {
    **_E7_WHO,
    "default": "in the default set",
    "raw_depth": "raw sequencing depth",
    "callable_bases": "bases the strict mask leaves callable",
    "het_sites": "heterozygous SNVs called",
    "het_per_callable_kb": "heterozygous SNVs per callable kb",
    "partly_callable_bin_fraction": "share of called bins with some bases missing",
})
_add("E7_route_shift", "E7", "sample of the default set", {
    **_E7_WHO,
    "theta_rel_vcf_vs_psmcfa": "relative theta difference, VCF fit (callable bases per bin) vs psmcfa fit",
    "mean_log10_ne_vcf_vs_psmcfa": "mean log10 Ne difference in the window, VCF vs psmcfa",
    "partly_callable_bin_fraction": "share of called bins with some bases missing",
    "missing_share_of_called_bins": "share of the called bins' bases that are missing",
})
_add("E7_split", "E7", "sample of the split-first-window check", {
    **_E7_WHO,
    "log10_first_vs_next_default": "log10 of the first window's Ne over the next one's, default pattern",
    "log10_first_vs_next_split": "the same, first window split: its four intervals' log10 Ne averaged by duration",
    "log10_first_vs_next_split_lo": "the same with the lowest of the four intervals",
    "log10_first_vs_next_split_hi": "the same with the highest of the four intervals",
    "max_abs_log10_ne_split_vs_default_per_interval": "largest per-interval |log10 Ne difference|, split vs default",
    "ne_next_split": "the next window's Ne, first window split",
})
_add("E7_fits", "E7", "sample and route", {
    "sample_id": "sample",
    "route": "rs-psmcfa, rs-vcf, c or rs-split",
    "n0": "N0 of the fit (Ne = lambda N0)",
})
_add("E7_curves", "E7", "sample, route and interval", {
    "sample_id": "sample",
    "route": "rs-psmcfa, rs-vcf, c or rs-split",
    "interval": "PSMC interval",
    "years": "its start (years ago)",
    "ne": "its Ne",
})
_add("E7_giab", "E7", "sample compared with GIAB", {
    "sample_id": "sample",
    "precision": "share of its heterozygous SNVs that GIAB has",
    "recall": "share of GIAB's heterozygous SNVs it has",
    "truth_het_snvs": "GIAB heterozygous SNVs in strict mask and GIAB's confident regions",
    "eval_bases": "bases compared",
})
_add("E7_throughput", "E7", "route and tool", {
    "route": "rs-psmcfa, rs-vcf, c, rs-split or boot-rs",
    "tool": "rs or c",
    "jobs": "runs (a fit each; a bootstrap run fits the data, then every replicate)",
    "threads": "threads per run",
    "wall_median_s": "median wall time per run",
    "core_hours": "CPU core-hours of all of them",
    "peak_rss_median_mb": "median peak memory",
})


def describe_all() -> str:
    """Every schema, as the figures README lists them."""
    return "\n\n".join(s.describe() for s in SCHEMAS.values())


__all__ = ["CURVE_COLUMNS", "SCHEMAS", "TableSchema", "describe_all"]
