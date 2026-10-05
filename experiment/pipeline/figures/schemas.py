"""The tables the figures read: file, columns and meaning.

This is the contract between the experiments and the figures. Each
experiment's ``report`` writes its tables to ``<runs>/<EXP>/tables/``; the
figures read nothing else (plus :mod:`..config`, e.g. the demographic
models, whose true Ne(t) the truth records hold too). A figure checks that
every column it needs is present (:func:`read`) and names the table and the
missing columns otherwise; extra columns are allowed.

Conventions, as in E1's tables: times in ``years`` (scaled with the
experiment's mu and g), sizes ``ne`` in diploid individuals, replicates and
datasets counted from 1, ``tool`` is ``c`` (C psmc), ``rs`` (psmc-rs 0.2)
or ``rs01`` (psmc-rs 0.1). A curve table has one row per PSMC interval:
``ne`` holds from ``years`` to the next row's ``years`` (the first row's
``years`` is 0, the last interval runs to infinity), exactly as
:meth:`..results.Fit.curve` gives it. Booleans are ``true``/``false``.

E1's tables exist (``experiments/e1.py``). The schemas of E2 to E6 are what
the figures expect; :mod:`.fake` writes small tables in these schemas for
developing and testing the figures until the experiments write theirs.
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


def _schema(experiment: str, file: str, rows: str, columns: Dict[str, str]) -> TableSchema:
    return TableSchema(experiment, file, tuple(columns.items()), rows)


SCHEMAS: Dict[str, TableSchema] = {}


def _add(key: str, schema: TableSchema) -> None:
    SCHEMAS[key] = schema


# ---------------------------------------------------------------------------
# E1 (written by experiments/e1.py; only the columns the figures use)
# ---------------------------------------------------------------------------

_add("E1_curves", _schema("E1", "E1_curves.csv", "model, replicate, tool and interval", {
    "model": "demographic model (config.MODELS)",
    "replicate": "replicate (1..)",
    "tool": "c or rs",
    **CURVE_COLUMNS,
}))
_add("E1_truth", _schema("E1", "E1_truth.csv", "model and time of a dense log grid", {
    "model": "demographic model",
    "years": "years ago",
    "ne": "true Ne at that time",
}))
_add("E1_table1", _schema("E1", "E1_table1.csv", "model", {
    "model": "demographic model",
    "window_lo_years": "start of the informative window (1% quantile of the true coalescence time)",
    "window_hi_years": "end of the informative window (99% quantile)",
}))
_add("E1_replicates", _schema("E1", "E1_replicates.csv", "model, replicate and tool", {
    "model": "demographic model",
    "replicate": "replicate",
    "tool": "c or rs",
    "rmse_log10_ne_vs_truth": "RMSE of log10 Ne against the truth on the informative window",
}))
_add("E1_rs_vs_c", _schema("E1", "E1_rs_vs_c.csv", "model and replicate", {
    "model": "demographic model",
    "replicate": "replicate",
    "rmse_log10_ne_rs_vs_c": "RMSE of log10 Ne between the tools on the informative window",
    "max_abs_log10_ne_diff": "largest |log10 Ne_rs - log10 Ne_c| on the window",
}))
_add("E1_traces", _schema("E1", "E1_traces.csv", "model, replicate, tool and EM iteration", {
    "model": "demographic model",
    "replicate": "replicate",
    "tool": "c or rs",
    "iteration": "completed EM iterations i (0 = the start)",
    "loglike": "log-likelihood after i iterations (empty for C at i = N_ITER: it prints none)",
    "theta": "theta per bin after i iterations",
    "rho": "rho per bin after i iterations",
}))

# ---------------------------------------------------------------------------
# E2: real genomes (Fig 3)
# ---------------------------------------------------------------------------

_add("E2_curves", _schema("E2", "E2_curves.csv", "sample, tool, replicate and interval", {
    "sample": "config.RealGenome.sample_id (HLemySub1, HLhydTec1, HLpelCas1, Papuan)",
    "tool": "c or rs",
    "replicate": "0: the fit of the whole input; 1..N: psmc-rs bootstrap replicates (tool rs)",
    **CURVE_COLUMNS,
}))
_add("E2_summary", _schema("E2", "E2_summary.csv", "sample", {
    "sample": "sample id",
    "mu": "mutation rate used for scaling",
    "gen_years": "generation time used for scaling",
    "rmse_log10_ne_rs_vs_c": "RMSE of log10 Ne between the tools' fits",
    "window_lo_years": "start of the window the RMSE is computed on",
    "window_hi_years": "end of that window",
    "bootstrap_replicates": "bootstrap replicates fitted (skipped ones excluded)",
}))

# ---------------------------------------------------------------------------
# E3: performance (Fig 4, Fig S4)
# ---------------------------------------------------------------------------

_add("E3_runs", _schema("E3", "E3_runs.csv", "tool, threads, genome size and repeat", {
    "tool": "c, rs01 or rs02",
    "threads": "threads given to the tool (1 for C psmc and psmc-rs 0.1)",
    "genome_mb": "simulated genome size (Mb)",
    "repeat": "repeat (1..)",
    "status": "ok, skipped or failed",
    "wall_seconds": "wall time of the whole run (empty unless ok)",
    "peak_rss_mb": "peak RSS of the process tree, MiB (empty unless ok)",
    "note": "why a run was skipped (e.g. estimated memory above the limit), else empty",
}))
_add("E3_threads", _schema("E3", "E3_threads.csv", "thread count and repeat (psmc-rs 0.2, thread-scaling genome)", {
    "genome_mb": "genome size (Mb)",
    "threads": "--threads",
    "repeat": "repeat (1..)",
    "wall_seconds": "wall time",
    "peak_rss_mb": "peak RSS of the process tree, MiB",
    "json_sha256": "SHA-256 of the output JSON",
}))

# ---------------------------------------------------------------------------
# E4: input preparation (Fig 5)
# ---------------------------------------------------------------------------

_add("E4_routes", _schema("E4", "E4_routes.csv", "replicate, mask and route", {
    "replicate": "replicate",
    "mask": "none (no missing data) or a config.MaskSpec name (clustered10, ...)",
    "structure": "none, clustered or dispersed",
    "missing_fraction": "fraction of uncallable positions (0 for none)",
    "route": "exact, fq2psmcfa, multihetsep or nomask",
    "theta_rel_bias": "theta / theta of the exact route on the same replicate without missing data - 1",
    "rmse_log10_ne_vs_truth": "RMSE of log10 Ne against the truth on the informative window",
    "bins_differ_fraction": "fraction of bins whose state (T, K, N) differs from the exact route's",
}))
_add("E4_curves", _schema("E4", "E4_curves.csv", "replicate, mask, route and interval", {
    "replicate": "replicate",
    "mask": "mask name or none",
    "route": "route",
    **CURVE_COLUMNS,
}))

# ---------------------------------------------------------------------------
# E5: bootstrap calibration (Fig 6)
# ---------------------------------------------------------------------------

_add("E5_calibration", _schema("E5", "E5_calibration.csv", "trunk size and time of the grid", {
    "trunk_bins": "--bootstrap-block-size (bins)",
    "years": "time of the informative-window grid",
    "sd_true": "SD of log10 Ne over the independent genomes' fits (true sampling SD)",
    "sd_boot_median": "median over datasets of the bootstrap SD of log10 Ne",
    "sd_boot_q25": "its 25% quantile over datasets",
    "sd_boot_q75": "its 75% quantile over datasets",
    "sd_ratio_median": "median over datasets of bootstrap SD / sd_true",
    "coverage": "fraction of datasets whose 95% percentile band contains the true Ne",
    "n_datasets": "bootstrapped datasets",
}))
_add("E5_curves", _schema("E5", "E5_curves.csv", "kind, dataset, trunk, replicate and interval", {
    "kind": "fit (one fit per independent genome) or bootstrap",
    "dataset": "genome (1..truth_replicates)",
    "trunk_bins": "trunk size of a bootstrap (0 for kind fit)",
    "replicate": "bootstrap replicate (0 for kind fit)",
    **CURVE_COLUMNS,
}))

# ---------------------------------------------------------------------------
# E6: supplementary analyses (Fig S2, S3)
# ---------------------------------------------------------------------------

_add("E6_formats", _schema("E6", "E6_formats.csv", "model and input format", {
    "model": "demographic model (E1 replicate config.E6.format_replicate)",
    "input": "psmcfa, vcf or multihetsep",
    "bins_differ": "bins whose state differs from the psmcfa's (over the common length)",
    "tail_bins_missing": "psmcfa bins after the last heterozygous site that the input lacks",
}))
_add("E6_format_curves", _schema("E6", "E6_format_curves.csv", "model, input format and interval", {
    "model": "demographic model",
    "input": "psmcfa, vcf or multihetsep",
    **CURVE_COLUMNS,
}))
_add("E6_tmrca_track", _schema("E6", "E6_tmrca_track.csv", "100 bp bin of the example window", {
    "start_bp": "0-based start of the bin on its chromosome",
    "true_tmrca_years": "true pairwise TMRCA at the bin's midpoint, in years",
    "post_mean_years": "posterior mean TMRCA",
    "post_map_years": "posterior MAP TMRCA",
    "q025_years": "2.5% posterior quantile",
    "q975_years": "97.5% posterior quantile",
}))
_add("E6_tmrca_hist2d", _schema("E6", "E6_tmrca_hist2d.csv", "estimator and cell of a log-log grid", {
    "estimator": "mean or map",
    "true_lo_years": "cell's lower edge, true TMRCA",
    "true_hi_years": "cell's upper edge, true TMRCA",
    "est_lo_years": "cell's lower edge, estimate",
    "est_hi_years": "cell's upper edge, estimate",
    "count": "bins in the cell",
}))
_add("E6_tmrca_calibration", _schema("E6", "E6_tmrca_calibration.csv", "class of true TMRCA", {
    "true_lo_years": "lower edge of the class",
    "true_hi_years": "upper edge of the class",
    "bins": "bins in the class",
    "coverage": "fraction whose 95% posterior interval contains the true TMRCA",
}))
_add("E6_tmrca_summary", _schema("E6", "E6_tmrca_summary.csv", "estimator", {
    "estimator": "mean or map",
    "pearson_log10": "Pearson correlation of log10 estimate and log10 true TMRCA over every bin",
    "coverage": "fraction of bins whose 95% interval contains the true TMRCA",
    "bins": "bins compared",
}))


def describe_all() -> str:
    """Every schema, as the figures README lists them."""
    return "\n\n".join(s.describe() for s in SCHEMAS.values())


__all__ = ["CURVE_COLUMNS", "SCHEMAS", "TableSchema", "describe_all"]
