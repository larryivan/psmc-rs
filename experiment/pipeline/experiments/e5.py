"""E5: bootstrap calibration (Fig 6).

How well does psmc-rs's bootstrap, which follows C psmc's, measure the
uncertainty of a PSMC curve? The runner simulates ``truth_replicates``
independent genomes of one model (:class:`.config.E5`) and fits each once
with psmc-rs 0.2: their spread at each time is the true sampling
distribution of log10 Ne(t). The first ``bootstrap_datasets`` genomes are
then bootstrapped by psmc-rs 0.2 at each trunk size, exactly as a user runs
it (``psmc-rs IN OUT N_ITER --bootstrap N --bootstrap-block-size B``: trunks
drawn with replacement up to the genome's length, every replicate fitted
from a fresh start with the settings of the fits). Layout under
``<runs>/E5/``::

    sims/<model>/repNN/genome.psmcfa                 simulated genome (not committed)
    truth/<model>/repNN.json                         seeds, counts, checksums, true Ne(t)
    fits/<model>/repNN.rs.json                       the genome fitted once (psmc-rs 0.2)
    boot/trunk<B>/repNN.rs.json                      the bootstrap run's main fit
    boot/trunk<B>/repNN/replicate_NNN.json           every replicate's parameters
    boot/trunk<B>/repNN/summary.json, summary.tsv    psmc-rs's band at the main curve's times
    boot/trunk<B>/repNN/replicate_NNN.psmc           every replicate's rounds (not committed)
    provenance/<job>.json                            one record per job
    tables/E5_*.csv                                  written by ``report E5``
    figures/fig6_bootstrap_calibration.{pdf,svg,png} Fig 6, drawn from the tables (experiment FIG)
    logs/                                            runner and job logs (not committed)

Every curve is read at the same times: the model's informative-window grid
(:class:`.config.TimeGrid`, as E1) and, for robustness, the fixed window.
At each time ``t``:

* the true sampling SD is the SD (n - 1) of log10 Ne(t) over the fits of the
  ``truth_replicates`` genomes, with a 95% interval from the chi-square
  distribution (which assumes normality);
* a bootstrap's SD is the SD of log10 Ne(t) over its replicates, and its
  ``band`` is the percentile band of the replicates' Ne(t), computed as
  psmc-rs computes it (linear interpolation between order statistics);
* the SD ratio of a dataset is its bootstrap SD over the true sampling SD;
  the table gives its median and spread over the datasets;
* the coverage is the share of datasets whose band holds the true Ne(t),
  with a Clopper-Pearson 95% interval; next to it, the share whose band holds
  the geometric mean of the fits (the estimator's expectation, which differs
  from the truth by PSMC's bias: the bootstrap can see the spread of the
  estimate but not its bias), and the share whose band holds the dataset's
  pattern-projected truth (:func:`.metrics.projected_truth` of its fit: the
  true Ne averaged over each free lambda's span, the best the pattern's
  steps can show), which takes the pattern's own error out of the bias.

The report also checks its inputs: each fit read the simulated genome (bins
and heterozygous bins of the truth record), each bootstrap ran with its seed,
trunk size and replicate count, and the band recomputed from the replicate
files equals psmc-rs's own summary. It records, without failing, whether the
main fit of each bootstrap run equals the genome's own fit (psmc-rs's results
do not depend on the run).

Tables (``tables/``):

* ``E5_summary.csv``: per trunk size and window, the headline numbers,
  with 95% intervals for the window-averaged coverage and the median SD
  ratio from resampling the datasets (the independent units; the points of
  one dataset are strongly correlated), and the spread of each dataset's
  share of covered points.
* ``E5_sd_ratio.csv``: per trunk size and time, the true sampling SD, the
  bootstrap SDs and their ratio (median, quartiles and range over datasets).
* ``E5_coverage.csv``: per trunk size and time, the pointwise coverage of
  the truth (and of the fits' mean) by the band, and the band widths.
* ``E5_sampling.csv``: per time, the sampling distribution of the fits.
* ``E5_bands.csv``: per dataset, trunk size and time, the fit and its band.
* ``E5_datasets.csv``: per dataset and trunk size, the bootstrap run: its
  replicates, trunks, seed, checks, wall time and peak RSS.
* ``E5_fits.csv``: per genome, the fit: theta, rho, t_max, N0,
  log-likelihood, RMSE against the truth, wall time and peak RSS, and the
  genome's longest stretch without a heterozygous site (an outlier's
  likely cause: a long segment of very recent coalescence).
* ``E5_seeds.csv``: every seed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import rep, run_record, write_csv

NAME = "E5"
TABLES = {
    "summary": "E5_summary.csv",
    "sd_ratio": "E5_sd_ratio.csv",
    "coverage": "E5_coverage.csv",
    "sampling": "E5_sampling.csv",
    "bands": "E5_bands.csv",
    "datasets": "E5_datasets.csv",
    "fits": "E5_fits.csv",
    "seeds": "E5_seeds.csv",
}
WINDOWS = ("informative", "fixed")
SUMMARY_CHECK_RTOL = 1e-9  # recomputed band vs psmc-rs's summary.json (full precision)
MAIN_FIT_KEYS = ("theta", "rho", "t_max", "lam", "loglike")


def trunk_dir(trunk_bins: int) -> str:
    return f"trunk{trunk_bins}"


def bootstrap_stage(trunk_bins: int) -> str:
    """The seed stage of a bootstrap run (see :func:`.config.seed`)."""
    return f"bootstrap/{trunk_dir(trunk_bins)}"


def bootstrap_seed(cfg: config.E5, r: int, trunk_bins: int) -> int:
    return config.seed(NAME, cfg.model, r, bootstrap_stage(trunk_bins))


def resampling_stage(window: str, trunk_bins: int) -> str:
    """The seed stage (replicate 0) of the datasets' resampling for one window and trunk size."""
    return f"summary/{window}/{trunk_dir(trunk_bins)}"


@dataclass(frozen=True)
class Layout:
    """Where E5's files are."""

    root: Path
    model: str

    def sim_dir(self, r: int) -> Path:
        return self.root / "sims" / self.model / rep(r)

    def psmcfa(self, r: int) -> Path:
        return self.sim_dir(r) / simulate.PSMCFA_NAME

    def truth(self, r: int) -> Path:
        return self.root / "truth" / self.model / f"{rep(r)}.json"

    def fit_json(self, r: int) -> Path:
        return self.root / "fits" / self.model / f"{rep(r)}.rs.json"

    def boot_json(self, r: int, trunk_bins: int) -> Path:
        return self.root / "boot" / trunk_dir(trunk_bins) / f"{rep(r)}.rs.json"

    def boot_dir(self, r: int, trunk_bins: int) -> Path:
        return self.root / "boot" / trunk_dir(trunk_bins) / rep(r)

    def boot_summary(self, r: int, trunk_bins: int) -> Path:
        return self.boot_dir(r, trunk_bins) / "summary.json"

    def boot_summary_tsv(self, r: int, trunk_bins: int) -> Path:
        return self.boot_dir(r, trunk_bins) / "summary.tsv"

    def replicate_json(self, r: int, trunk_bins: int, i: int) -> Path:
        return self.boot_dir(r, trunk_bins) / f"replicate_{i:03d}.json"

    def record(self, job_name: str) -> Path:
        return self.root / "provenance" / f"{job_name}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_name(kind: str, model: str, r: int, trunk_bins: Optional[int] = None) -> str:
    """``sim/<model>/repNN``, ``fit/<model>/repNN`` or ``boot-trunk<B>/<model>/repNN``."""
    k = f"{kind}-{trunk_dir(trunk_bins)}" if trunk_bins is not None else kind
    return f"{k}/{model}/{rep(r)}"


def layout(ctx: Context) -> Layout:
    return Layout(ctx.exp_dir(NAME), ctx.profile.e5.model)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def bootstrap_seconds(costs: config.Costs, cfg: config.E5, genome_mb: float) -> float:
    """Expected wall time of one bootstrap run: its main fit, then the replicates in waves of ``threads``."""
    threads = cfg.bootstrap_threads
    main = costs.rs02_seconds(genome_mb, cfg.inference, threads)
    return main + math.ceil(cfg.bootstrap_replicates / threads) * costs.rs02_seconds(genome_mb, cfg.inference)


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e5
    costs = ctx.profile.costs
    lay = layout(ctx)
    model = cfg.model
    if cfg.bootstrap_datasets > cfg.truth_replicates:
        raise ValueError("E5: bootstrap_datasets must not exceed truth_replicates")
    genome_mb = cfg.genome.length / 1e6
    inference, scaling = asdict(cfg.inference), asdict(cfg.scaling)
    sim_tools = {
        "pipeline": provenance.python_identity(
            ["simulate.py", "demography.py"], ["msprime", "tskit", "numpy", "stdpopsim"]
        ),
    }
    rs = ctx.rs02()
    rs_ident = {rs.name: rs.identity}
    out: List[Job] = []
    for r in range(1, cfg.truth_replicates + 1):
        psmcfa = lay.psmcfa(r)
        out.append(
            Job(
                id=f"{NAME}/{job_name('sim', model, r)}",
                command=ctx.pipeline_cmd(
                    "simulate", "--profile", ctx.profile.name, "--experiment", NAME, "--model", model,
                    "--replicate", str(r), "--out-dir", str(lay.sim_dir(r)), "--truth", str(lay.truth(r)),
                    "--no-vcf",
                ),
                outputs=[psmcfa, lay.truth(r)],
                mem_gb=costs.simulate_gb,
                est_seconds=costs.simulate_s_base + costs.simulate_s_per_mb * genome_mb,
                settings={
                    "model": asdict(config.MODELS[model]),
                    "genome": asdict(cfg.genome),
                    "simulation": asdict(cfg.simulation),
                },
                tools=sim_tools,
                seeds={c: simulate.seeds_for(NAME, model, r, c) for c in cfg.genome.names()},
                description=f"simulate {model} genome {r}",
            )
        )
        out.append(
            Job(
                id=f"{NAME}/{job_name('fit', model, r)}",
                command=tools.rs02_fit(rs, cfg.inference, cfg.scaling, psmcfa, lay.fit_json(r), threads=cfg.rs_threads),
                inputs=[psmcfa],
                outputs=[lay.fit_json(r)],
                cores=cfg.rs_threads,
                mem_gb=costs.rs02_gb,
                est_seconds=costs.rs02_seconds(genome_mb, cfg.inference, cfg.rs_threads),
                settings={"inference": inference, "scaling": scaling, "threads": cfg.rs_threads},
                tools=rs_ident,
                description=f"psmc-rs 0.2 on {model} genome {r}",
            )
        )
        if r > cfg.bootstrap_datasets:
            continue
        for trunk in cfg.trunk_bins:
            seed = bootstrap_seed(cfg, r, trunk)
            boot = {"replicates": cfg.bootstrap_replicates, "trunk_bins": trunk, "seed": seed,
                    "iterations": cfg.inference.n_iter}
            out.append(
                Job(
                    id=f"{NAME}/{job_name('boot', model, r, trunk)}",
                    command=tools.rs02_fit(
                        rs, cfg.inference, cfg.scaling, psmcfa, lay.boot_json(r, trunk),
                        threads=cfg.bootstrap_threads,
                        extra=[
                            "--bootstrap", str(cfg.bootstrap_replicates),
                            "--bootstrap-block-size", str(trunk),
                            "--bootstrap-seed", str(seed),
                            "--bootstrap-dir", str(lay.boot_dir(r, trunk)),
                        ],
                    ),
                    inputs=[psmcfa],
                    # The replicate files are not declared: a replicate whose
                    # fresh start is invalid is skipped (summary.json lists it),
                    # and the report reads exactly the ones the summary names.
                    outputs=[lay.boot_json(r, trunk), lay.boot_summary(r, trunk), lay.boot_summary_tsv(r, trunk)],
                    cores=cfg.bootstrap_threads,
                    mem_gb=round(costs.rs02_gb * cfg.bootstrap_threads, 2),
                    est_seconds=bootstrap_seconds(costs, cfg, genome_mb),
                    settings={"inference": inference, "scaling": scaling, "threads": cfg.bootstrap_threads,
                              "bootstrap": boot},
                    tools=rs_ident,
                    seeds={"bootstrap": seed},
                    description=f"psmc-rs 0.2 bootstrap of {model} genome {r}, trunks of {trunk} bins",
                )
            )
    out.append(report_job(ctx, lay))
    return out


def report_inputs(cfg: config.E5, lay: Layout) -> List[Path]:
    """Every file the tables are made from."""
    files: List[Path] = []
    for r in range(1, cfg.truth_replicates + 1):
        files += [lay.truth(r), lay.fit_json(r), lay.record(job_name("fit", cfg.model, r))]
        if r <= cfg.bootstrap_datasets:
            for trunk in cfg.trunk_bins:
                files += [lay.boot_json(r, trunk), lay.boot_summary(r, trunk),
                          lay.record(job_name("boot", cfg.model, r, trunk))]
    return files


def report_job(ctx: Context, lay: Layout) -> Job:
    cfg, costs = ctx.profile.e5, ctx.profile.costs
    return Job(
        id=f"{NAME}/report",
        command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
        inputs=report_inputs(cfg, lay),
        outputs=[lay.table(k) for k in TABLES],
        mem_gb=costs.report_gb,
        est_seconds=costs.report_s,
        settings={"e5": asdict(cfg)},
        tools={"pipeline": provenance.python_identity(
            ["experiments/e5.py", "experiments/e1.py", "results.py", "metrics.py", "demography.py"],
            ["numpy", "scipy"],
        )},
        description="E5 tables",
    )


# ---------------------------------------------------------------------------
# Reading the outputs back
# ---------------------------------------------------------------------------


def trunk_count(sequence_bins: Sequence[int], trunk_bins: int) -> int:
    """How many trunks splitfa (and psmc-rs) cut sequences of these lengths into.

    Each sequence is cut into trunks of ``trunk_bins``; a remainder shorter
    than 1.5 trunks stays in the last one (psmc-rs ``trunk_ranges``).
    """
    if trunk_bins <= 0:
        raise ValueError("trunk_bins must be positive")
    tail = trunk_bins * 3 // 2
    n = 0
    for length in sequence_bins:
        start = 0
        while start < length:
            start = length if length - start < tail else start + trunk_bins
            n += 1
    return n


@dataclass
class GenomeFit:
    """One simulated genome, its truth record and its fit."""

    replicate: int
    truth: Dict
    params: Dict
    fit: results.Fit
    curve: results.Curve
    run: Dict


@dataclass
class Bootstrap:
    """One bootstrap run: its main fit, psmc-rs's summary and the replicates' curves."""

    main: Dict
    summary: Dict
    numbers: List[int]  # replicate numbers (from 1) that were fitted
    curves: List[results.Curve]
    run: Dict


def check_input(params: Dict, truth: Dict, what: str) -> None:
    """Raise unless the fit read the genome of the truth record (bins and heterozygous bins)."""
    bins = sum(ch["bins"] for ch in truth["chromosomes"])
    het = sum(ch["het_bins"] for ch in truth["chromosomes"])
    inp = params.get("input", {})
    if inp.get("bins") != bins or inp.get("het_bins") != het:
        raise ValueError(
            f"{what}: the fit did not read the simulated genome (truth {bins} bins/{het} het, "
            f"fit {inp.get('bins')}/{inp.get('het_bins')})"
        )


def load_genome(lay: Layout, cfg: config.E5, r: int) -> GenomeFit:
    truth = json.loads(lay.truth(r).read_text())
    params = results.read_rs_json(lay.fit_json(r))
    if int(params.get("n_iter", -1)) != cfg.inference.n_iter:
        raise ValueError(f"{lay.fit_json(r)}: not a fit of {cfg.inference.n_iter} iterations")
    check_input(params, truth, f"{cfg.model} {rep(r)}")
    fit = results.fit_from_rs(params)
    sc = cfg.scaling
    curve = fit.curve(sc.mu, sc.gen_years, cfg.inference.bin_size)
    return GenomeFit(r, truth, params, fit, curve, run_record(lay.record(job_name("fit", cfg.model, r))))


def load_bootstrap(lay: Layout, cfg: config.E5, r: int, trunk: int, truth: Dict) -> Bootstrap:
    """Read one bootstrap run and check it is the run the configuration asks for."""
    what = f"{cfg.model} {rep(r)} {trunk_dir(trunk)}"
    main = results.read_rs_json(lay.boot_json(r, trunk))
    if int(main.get("n_iter", -1)) != cfg.inference.n_iter:
        raise ValueError(f"{lay.boot_json(r, trunk)}: not a fit of {cfg.inference.n_iter} iterations")
    check_input(main, truth, what)
    summary = json.loads(lay.boot_summary(r, trunk).read_text())
    seed = bootstrap_seed(cfg, r, trunk)
    if summary.get("block_size") != trunk or summary.get("seed") != seed:
        raise ValueError(
            f"{what}: summary.json has trunks of {summary.get('block_size')} bins and seed "
            f"{summary.get('seed')}, expected {trunk} and {seed}"
        )
    skipped = [int(i) for i in summary.get("skipped_replicates", [])]
    numbers = [i for i in range(1, cfg.bootstrap_replicates + 1) if i not in skipped]
    if len(numbers) != int(summary.get("n_replicates", -1)):
        raise ValueError(
            f"{what}: summary.json counts {summary.get('n_replicates')} replicates and skips {skipped}, "
            f"but the run asked for {cfg.bootstrap_replicates}"
        )
    if not numbers:
        raise ValueError(f"{what}: no replicate was fitted")
    sc = cfg.scaling
    curves = []
    for i in numbers:
        params = results.read_rs_json(lay.replicate_json(r, trunk, i))
        curves.append(results.fit_from_rs(params).curve(sc.mu, sc.gen_years, cfg.inference.bin_size))
    run = run_record(lay.record(job_name("boot", cfg.model, r, trunk)))
    return Bootstrap(main, summary, numbers, curves, run)


# ---------------------------------------------------------------------------
# Statistics (numpy arrays: rows are fits or replicates, columns grid times)
# ---------------------------------------------------------------------------


def quantile(values, q: float, axis: int = 0):
    """Quantiles by linear interpolation between order statistics, as psmc-rs computes its band."""
    import numpy as np

    return np.quantile(np.asarray(values, dtype=float), q, axis=axis, method="linear")


def iqr(values, axis: int = 0):
    return quantile(values, 0.75, axis) - quantile(values, 0.25, axis)


def sd_interval(sd, n: int, level: float = 0.95) -> Tuple:
    """A chi-square ``level`` interval for a standard deviation estimated from ``n`` normal values."""
    import numpy as np
    from scipy import stats

    a = (1.0 - level) / 2.0
    df = n - 1
    sd = np.asarray(sd, dtype=float)
    return sd * math.sqrt(df / stats.chi2.ppf(1.0 - a, df)), sd * math.sqrt(df / stats.chi2.ppf(a, df))


def clopper_pearson(k, n: int, level: float = 0.95) -> Tuple:
    """Exact binomial ``level`` interval for ``k`` successes out of ``n`` (arrays allowed)."""
    import numpy as np
    from scipy import stats

    k = np.asarray(k, dtype=float)
    a = (1.0 - level) / 2.0
    lo = np.where(k > 0, stats.beta.ppf(a, k, n - k + 1), 0.0)
    hi = np.where(k < n, stats.beta.ppf(1.0 - a, k + 1, n - k), 1.0)
    return lo, hi


def sampling_stats(fits_ne, truth_ne, band: Tuple[float, float]) -> Dict:
    """The sampling distribution of the fits at each time (``fits_ne``: fits x times)."""
    import numpy as np

    ne = np.asarray(fits_ne, dtype=float)
    logs = np.log10(ne)
    log_truth = np.log10(np.asarray(truth_ne, dtype=float))
    n = ne.shape[0]
    if n < 2:
        raise ValueError("the sampling SD needs at least two fits")
    sd = logs.std(axis=0, ddof=1)
    sd_lo, sd_hi = sd_interval(sd, n)
    mean = logs.mean(axis=0)
    return {
        "n": n,
        "mean_log10": mean,
        "median_ne": quantile(ne, 0.5),
        "lo_ne": quantile(ne, band[0]),
        "hi_ne": quantile(ne, band[1]),
        "sd_log10": sd,
        "sd_lo": sd_lo,
        "sd_hi": sd_hi,
        "iqr_log10": iqr(logs),
        "bias_log10": mean - log_truth,
        "rmse_log10": np.sqrt(np.mean((logs - log_truth) ** 2, axis=0)),
    }


def band_stats(replicates_ne, band: Tuple[float, float]) -> Dict:
    """One bootstrap at each time (``replicates_ne``: replicates x times): its band and spread."""
    import numpy as np

    ne = np.asarray(replicates_ne, dtype=float)
    logs = np.log10(ne)
    return {
        "lo_ne": quantile(ne, band[0]),
        "median_ne": quantile(ne, 0.5),
        "hi_ne": quantile(ne, band[1]),
        "sd_log10": logs.std(axis=0, ddof=1) if ne.shape[0] > 1 else np.zeros(ne.shape[1]),
        "iqr_log10": iqr(logs),
    }


def calibration(boots: Sequence[Dict], sampling: Dict, truth_ne, projected_ne=None) -> Dict:
    """SD ratios and coverage of the bootstraps of one trunk size, at each time.

    ``boots`` are :func:`band_stats` of the datasets; ``sampling`` is
    :func:`sampling_stats` of the fits; ``projected_ne`` (datasets x times),
    when given, is each dataset's pattern-projected truth
    (:func:`.metrics.projected_truth` of its fit), whose coverage is
    reported next to the coverage of the true Ne.
    """
    import numpy as np

    truth_ne = np.asarray(truth_ne, dtype=float)
    lo = np.array([b["lo_ne"] for b in boots])
    hi = np.array([b["hi_ne"] for b in boots])
    sd = np.array([b["sd_log10"] for b in boots])
    iq = np.array([b["iqr_log10"] for b in boots])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sd / sampling["sd_log10"]
        iqr_ratio = iq / sampling["iqr_log10"]
    covers = (lo <= truth_ne) & (truth_ne <= hi)
    mean_ne = 10.0 ** sampling["mean_log10"]
    covers_mean = (lo <= mean_ne) & (mean_ne <= hi)
    n = len(boots)
    k = covers.sum(axis=0)
    ci_lo, ci_hi = clopper_pearson(k, n)
    projected = {}
    if projected_ne is not None:
        proj = np.asarray(projected_ne, dtype=float)
        covers_proj = (lo <= proj) & (proj <= hi)
        k_proj = covers_proj.sum(axis=0)
        p_lo, p_hi = clopper_pearson(k_proj, n)
        projected = {
            "covers_projected": covers_proj,  # datasets x times
            "covered_projected": k_proj,
            "coverage_projected": k_proj / n,
            "coverage_projected_lo": p_lo,
            "coverage_projected_hi": p_hi,
        }
    return {
        "datasets": n,
        "ratio": ratio,  # datasets x times
        "ratio_median": np.median(ratio, axis=0),
        "ratio_q25": quantile(ratio, 0.25),
        "ratio_q75": quantile(ratio, 0.75),
        "ratio_min": ratio.min(axis=0),
        "ratio_max": ratio.max(axis=0),
        "iqr_ratio_median": np.median(iqr_ratio, axis=0),
        "sd_boot_median": np.median(sd, axis=0),
        "covers": covers,  # datasets x times
        "covered": k,
        "coverage": k / n,
        "coverage_lo": ci_lo,
        "coverage_hi": ci_hi,
        "covered_mean": covers_mean.sum(axis=0),
        "coverage_mean": covers_mean.mean(axis=0),
        "band_width_log10": np.log10(hi / lo),  # datasets x times
        "band_width_log10_median": np.median(np.log10(hi / lo), axis=0),
        **projected,
    }


def summary_check(boot: Bootstrap) -> float:
    """Largest relative difference between psmc-rs's summary band and the band recomputed from the replicates.

    psmc-rs reads each replicate's Ne at the start times of the main curve
    (``x_years``) and takes linear quantiles of Ne; this does the same with
    the replicate files.
    """
    import numpy as np

    points = boot.summary["points"]
    x = np.array([p["x_years"] for p in points])
    ne = np.array([c.at(x) for c in boot.curves])
    worst = 0.0
    for key, q in zip(("ne_q025", "ne_q500", "ne_q975"), config.PSMC_RS_SUMMARY_QUANTILES):
        mine = quantile(ne, q)
        theirs = np.array([p[key] for p in points])
        worst = max(worst, float(np.max(np.abs(mine / theirs - 1.0))))
    return worst


def main_fit_difference(main: Dict, own: Dict) -> float:
    """Largest absolute difference between the parameters and log-likelihoods of two fits (0 if identical)."""
    worst = 0.0
    for key in MAIN_FIT_KEYS:
        a, b = main.get(key), own.get(key)
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        if len(a) != len(b) or any(x is None for x in a + b):
            return math.inf
        worst = max([worst, *(abs(float(x) - float(y)) for x, y in zip(a, b))])
    return worst


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def window_grids(epochs: Sequence[demography.Epoch], cfg: config.E5) -> Dict[str, object]:
    g = cfg.scaling.gen_years
    return {"informative": metrics.model_grid(epochs, cfg.grid, g), "fixed": metrics.fixed_grid(cfg.grid)}


def analyse(genomes: List[GenomeFit], boots: Dict[Tuple[int, int], Bootstrap], cfg: config.E5, grid, epochs) -> Dict:
    """Sampling distribution, bootstrap bands and calibration on one grid of times."""
    import numpy as np

    sc = cfg.scaling
    truth_ne = metrics.truth_on_grid(epochs, grid, sc.gen_years)
    fits_ne = np.array([gn.curve.at(grid) for gn in genomes])
    projected_ne = np.array([
        metrics.projected_truth(gn.fit, epochs, sc.mu, sc.gen_years, cfg.inference.bin_size).at(grid)
        for gn in genomes[:cfg.bootstrap_datasets]
    ])
    sampling = sampling_stats(fits_ne, truth_ne, cfg.band)
    datasets = range(1, cfg.bootstrap_datasets + 1)
    bands = {(d, trunk): band_stats(np.array([c.at(grid) for c in boots[(d, trunk)].curves]), cfg.band)
             for trunk in cfg.trunk_bins for d in datasets}
    by_trunk = {trunk: calibration([bands[(d, trunk)] for d in datasets], sampling, truth_ne, projected_ne)
                for trunk in cfg.trunk_bins}
    return {"grid": grid, "truth_ne": truth_ne, "fits_ne": fits_ne, "projected_ne": projected_ne,
            "sampling": sampling, "bands": bands, "calibration": by_trunk}


def dataset_resampling(c: Dict, resamples: int, seed: int, level: float = 0.95) -> Dict[str, float]:
    """Percentile intervals over datasets resampled with replacement, for one trunk size and window.

    Datasets are the independent units: the pointwise coverages and SD
    ratios of one dataset are strongly correlated along the window, so the
    window's summaries are resampled by dataset. Statistics: the
    window-averaged coverage (the mean over datasets of each dataset's share
    of covered points; of the projected truth too, with the same resamples)
    and the median over the window of the per-point median SD ratio.
    """
    import numpy as np

    covers = np.asarray(c["covers"], dtype=float)  # datasets x times
    ratio = np.asarray(c["ratio"], dtype=float)
    n = covers.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    share = covers.mean(axis=1)
    coverage = share[idx].mean(axis=1)
    ratio_median = np.array([np.median(np.median(ratio[i], axis=0)) for i in idx])
    a = (1.0 - level) / 2.0
    projected = {}
    if "covers_projected" in c:
        cov_proj = np.asarray(c["covers_projected"], dtype=float).mean(axis=1)[idx].mean(axis=1)
        projected = {
            "coverage_projected_mean_ci_lo": float(np.quantile(cov_proj, a)),
            "coverage_projected_mean_ci_hi": float(np.quantile(cov_proj, 1.0 - a)),
        }
    return {
        "coverage_dataset_sd": float(share.std(ddof=1)) if n > 1 else 0.0,
        "coverage_dataset_min": float(share.min()),
        "coverage_dataset_max": float(share.max()),
        "coverage_mean_ci_lo": float(np.quantile(coverage, a)),
        "coverage_mean_ci_hi": float(np.quantile(coverage, 1.0 - a)),
        "sd_ratio_median_ci_lo": float(np.quantile(ratio_median, a)),
        "sd_ratio_median_ci_hi": float(np.quantile(ratio_median, 1.0 - a)),
        **projected,
    }


def summary_rows(cfg: config.E5, window: str, res: Dict, trunks: Dict[int, int]) -> List[Dict]:
    import numpy as np

    grid, s = res["grid"], res["sampling"]
    rows = []
    for trunk in cfg.trunk_bins:
        c = res["calibration"][trunk]
        rm = c["ratio_median"]
        seed = config.seed(NAME, cfg.model, 0, resampling_stage(window, trunk))
        resampled = dataset_resampling(c, cfg.dataset_resamples, seed)
        rows.append(
            {
                "trunk_bins": trunk,
                "trunk_mb": trunk * cfg.inference.bin_size / 1e6,
                "trunks": trunks[trunk],
                "window": window,
                "window_lo_years": float(grid[0]),
                "window_hi_years": float(grid[-1]),
                "points": len(grid),
                "fits": s["n"],
                "datasets": c["datasets"],
                "replicates_per_dataset": cfg.bootstrap_replicates,
                "sd_ratio_median": float(np.median(rm)),
                "sd_ratio_median_ci_lo": resampled["sd_ratio_median_ci_lo"],
                "sd_ratio_median_ci_hi": resampled["sd_ratio_median_ci_hi"],
                "sd_ratio_min": float(rm.min()),
                "sd_ratio_max": float(rm.max()),
                "share_points_sd_ratio_below_1": float(np.mean(rm < 1.0)),
                "iqr_ratio_median": float(np.median(c["iqr_ratio_median"])),
                "coverage_nominal": cfg.band[1] - cfg.band[0],
                "coverage_mean": float(np.mean(c["coverage"])),
                "coverage_mean_ci_lo": resampled["coverage_mean_ci_lo"],
                "coverage_mean_ci_hi": resampled["coverage_mean_ci_hi"],
                "coverage_dataset_sd": resampled["coverage_dataset_sd"],
                "coverage_dataset_min": resampled["coverage_dataset_min"],
                "coverage_dataset_max": resampled["coverage_dataset_max"],
                "coverage_min": float(np.min(c["coverage"])),
                "coverage_of_fit_mean_mean": float(np.mean(c["coverage_mean"])),
                "coverage_projected_mean": float(np.mean(c["coverage_projected"])),
                "coverage_projected_mean_ci_lo": resampled["coverage_projected_mean_ci_lo"],
                "coverage_projected_mean_ci_hi": resampled["coverage_projected_mean_ci_hi"],
                "coverage_projected_min": float(np.min(c["coverage_projected"])),
                "band_width_log10_median": float(np.median(c["band_width_log10"])),
                "sampling_width_log10_median": float(np.median(np.log10(s["hi_ne"] / s["lo_ne"]))),
                "sd_true_log10_median": float(np.median(s["sd_log10"])),
                "abs_bias_log10_median": float(np.median(np.abs(s["bias_log10"]))),
                "rmse_log10_median": float(np.median(s["rmse_log10"])),
            }
        )
    return rows


def per_time_rows(cfg: config.E5, res: Dict) -> Dict[str, List[Dict]]:
    grid, truth, s = res["grid"], res["truth_ne"], res["sampling"]
    out: Dict[str, List[Dict]] = {"sampling": [], "sd_ratio": [], "coverage": [], "bands": []}
    for k, t in enumerate(grid):
        out["sampling"].append(
            {
                "point": k, "years": float(t), "true_ne": float(truth[k]), "fits": s["n"],
                "mean_log10_ne": float(s["mean_log10"][k]), "median_ne": float(s["median_ne"][k]),
                "lo_ne": float(s["lo_ne"][k]), "hi_ne": float(s["hi_ne"][k]),
                "sd_log10": float(s["sd_log10"][k]), "sd_log10_ci_lo": float(s["sd_lo"][k]),
                "sd_log10_ci_hi": float(s["sd_hi"][k]), "iqr_log10": float(s["iqr_log10"][k]),
                "bias_log10": float(s["bias_log10"][k]), "rmse_log10": float(s["rmse_log10"][k]),
            }
        )
    for trunk in cfg.trunk_bins:
        c = res["calibration"][trunk]
        for k, t in enumerate(grid):
            base = {"trunk_bins": trunk, "point": k, "years": float(t), "true_ne": float(truth[k])}
            out["sd_ratio"].append(
                {
                    **base,
                    "sd_true_log10": float(s["sd_log10"][k]),
                    "sd_true_ci_lo": float(s["sd_lo"][k]),
                    "sd_true_ci_hi": float(s["sd_hi"][k]),
                    "sd_boot_log10_median": float(c["sd_boot_median"][k]),
                    "ratio_median": float(c["ratio_median"][k]),
                    "ratio_q25": float(c["ratio_q25"][k]),
                    "ratio_q75": float(c["ratio_q75"][k]),
                    "ratio_min": float(c["ratio_min"][k]),
                    "ratio_max": float(c["ratio_max"][k]),
                    "iqr_ratio_median": float(c["iqr_ratio_median"][k]),
                    "datasets": c["datasets"],
                }
            )
            out["coverage"].append(
                {
                    **base,
                    "datasets": c["datasets"],
                    "covered": int(c["covered"][k]),
                    "coverage": float(c["coverage"][k]),
                    "coverage_ci_lo": float(c["coverage_lo"][k]),
                    "coverage_ci_hi": float(c["coverage_hi"][k]),
                    "covered_fit_mean": int(c["covered_mean"][k]),
                    "coverage_fit_mean": float(c["coverage_mean"][k]),
                    "covered_projected": int(c["covered_projected"][k]),
                    "coverage_projected": float(c["coverage_projected"][k]),
                    "coverage_projected_ci_lo": float(c["coverage_projected_lo"][k]),
                    "coverage_projected_ci_hi": float(c["coverage_projected_hi"][k]),
                    "band_width_log10_median": float(c["band_width_log10_median"][k]),
                    "sampling_width_log10": float(math.log10(s["hi_ne"][k] / s["lo_ne"][k])),
                }
            )
        for d in range(1, cfg.bootstrap_datasets + 1):
            b = res["bands"][(d, trunk)]
            fit_ne, proj_ne = res["fits_ne"][d - 1], res["projected_ne"][d - 1]
            for k, t in enumerate(grid):
                out["bands"].append(
                    {
                        "dataset": d, "trunk_bins": trunk, "point": k, "years": float(t),
                        "true_ne": float(truth[k]), "fit_ne": float(fit_ne[k]),
                        "boot_lo_ne": float(b["lo_ne"][k]), "boot_median_ne": float(b["median_ne"][k]),
                        "boot_hi_ne": float(b["hi_ne"][k]), "boot_sd_log10": float(b["sd_log10"][k]),
                        "covers_truth": int(c["covers"][d - 1][k]),
                        "projected_ne": float(proj_ne[k]),
                        "covers_projected": int(c["covers_projected"][d - 1][k]),
                    }
                )
    return out


def dataset_rows(cfg: config.E5, res: Dict, genomes: List[GenomeFit], boots: Dict, trunks: Dict[int, int]) -> List[Dict]:
    import numpy as np

    rows = []
    for trunk in cfg.trunk_bins:
        c = res["calibration"][trunk]
        for d in range(1, cfg.bootstrap_datasets + 1):
            b: Bootstrap = boots[(d, trunk)]
            diff = main_fit_difference(b.main, genomes[d - 1].params)
            run = b.run
            rows.append(
                {
                    "dataset": d,
                    "trunk_bins": trunk,
                    "trunks": trunks[trunk],
                    "seed": b.summary.get("seed"),
                    "replicates": len(b.numbers),
                    "skipped_replicates": ";".join(str(i) for i in b.summary.get("skipped_replicates", [])),
                    "sd_ratio_median": float(np.median(c["ratio"][d - 1])),
                    "coverage": float(np.mean(c["covers"][d - 1])),
                    "coverage_projected": float(np.mean(c["covers_projected"][d - 1])),
                    "band_width_log10_median": float(np.median(c["band_width_log10"][d - 1])),
                    "main_fit_equals_fit": diff == 0.0,
                    "main_fit_max_abs_diff": diff,
                    "summary_band_max_rel_diff": summary_check(b),
                    "threads": cfg.bootstrap_threads,
                    "wall_seconds": run.get("wall_seconds"),
                    "user_seconds": run.get("user_seconds"),
                    "peak_rss_mb": run.get("peak_rss_mb"),
                    "loadavg_before_1m": (run.get("loadavg_before") or [None])[0],
                    "run_start": run.get("start"),
                }
            )
    return rows


def fit_rows(cfg: config.E5, genomes: List[GenomeFit], epochs, grids: Dict) -> List[Dict]:
    sc = cfg.scaling
    g = sc.gen_years
    rows = []
    for gn in genomes:
        fit, run = gn.fit, gn.run
        projected = metrics.projected_truth(fit, epochs, sc.mu, g, cfg.inference.bin_size)
        gaps = [(ch.get("longest_het_gap_bp") or 0, ch["name"]) for ch in gn.truth["chromosomes"]]
        gap, gap_chrom = max(gaps) if gaps else (None, None)
        loglikes = results.rs_loglikes(gn.params)
        rows.append(
            {
                "replicate": gn.replicate,
                "bins": sum(ch["bins"] for ch in gn.truth["chromosomes"]),
                "het_bins": sum(ch["het_bins"] for ch in gn.truth["chromosomes"]),
                "het_sites": gn.truth["totals"]["het_sites"],
                "longest_het_gap_bp": gap,
                "longest_het_gap_chromosome": gap_chrom,
                "theta": fit.theta,
                "rho": fit.rho,
                "t_max": fit.t_max,
                "n0": fit.n0(sc.mu, cfg.inference.bin_size),
                "loglike_final": loglikes[-1] if loglikes else None,
                "rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(gn.curve, epochs, grids["informative"], g),
                "rmse_log10_ne_vs_truth_fixed": metrics.rmse_vs_truth(gn.curve, epochs, grids["fixed"], g),
                "rmse_log10_ne_vs_projected_truth": metrics.rmse_vs_projected_truth(
                    gn.curve, projected, grids["informative"]
                ),
                "bootstrapped": gn.replicate <= cfg.bootstrap_datasets,
                "wall_seconds": run.get("wall_seconds"),
                "peak_rss_mb": run.get("peak_rss_mb"),
                "run_start": run.get("start"),
            }
        )
    return rows


def seed_rows(cfg: config.E5, genomes: List[GenomeFit]) -> List[Dict]:
    rows = []
    for gn in genomes:
        for ch in gn.truth["chromosomes"]:
            for stage in ("ancestry", "mutations"):
                rows.append({"replicate": gn.replicate, "stage": f"{stage}/{ch['name']}", "seed": ch["seeds"][stage]})
        if gn.replicate <= cfg.bootstrap_datasets:
            for trunk in cfg.trunk_bins:
                rows.append({"replicate": gn.replicate, "stage": bootstrap_stage(trunk),
                             "seed": bootstrap_seed(cfg, gn.replicate, trunk)})
    for window in WINDOWS:
        for trunk in cfg.trunk_bins:
            stage = resampling_stage(window, trunk)
            rows.append({"replicate": 0, "stage": stage, "seed": config.seed(NAME, cfg.model, 0, stage)})
    return rows


def load_all(lay: Layout, cfg: config.E5):
    """The fits of every genome, the bootstrap runs and the true epochs (checked to agree)."""
    genomes = [load_genome(lay, cfg, r) for r in range(1, cfg.truth_replicates + 1)]
    epochs_record = genomes[0].truth["truth"]
    for gn in genomes[1:]:
        if gn.truth["truth"]["epochs"] != epochs_record["epochs"]:
            raise ValueError(f"{rep(gn.replicate)}: a different true model than {rep(1)}")
    epochs = demography.epochs_from_record(epochs_record)
    boots = {
        (d, trunk): load_bootstrap(lay, cfg, d, trunk, genomes[d - 1].truth)
        for d in range(1, cfg.bootstrap_datasets + 1)
        for trunk in cfg.trunk_bins
    }
    return genomes, boots, epochs


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E5 table from the fits and bootstraps; returns the paths written."""
    cfg = ctx.profile.e5
    lay = layout(ctx)
    genomes, boots, epochs = load_all(lay, cfg)
    sequence_bins = [ch["bins"] for ch in genomes[0].truth["chromosomes"]]
    trunks = {t: trunk_count(sequence_bins, t) for t in cfg.trunk_bins}
    grids = window_grids(epochs, cfg)
    res = {w: analyse(genomes, boots, cfg, grids[w], epochs) for w in WINDOWS}
    datasets = dataset_rows(cfg, res["informative"], genomes, boots, trunks)
    bad = [r for r in datasets if r["summary_band_max_rel_diff"] > SUMMARY_CHECK_RTOL]
    if bad:
        raise ValueError(
            f"the band recomputed from the replicate files differs from psmc-rs's summary.json "
            f"(dataset {bad[0]['dataset']}, trunks of {bad[0]['trunk_bins']} bins: "
            f"{bad[0]['summary_band_max_rel_diff']:.3g})"
        )
    tables: Dict[str, List[Dict]] = {
        "summary": [row for w in WINDOWS for row in summary_rows(cfg, w, res[w], trunks)],
        **per_time_rows(cfg, res["informative"]),
        "datasets": datasets,
        "fits": fit_rows(cfg, genomes, epochs, grids),
        "seeds": seed_rows(cfg, genomes),
    }
    for row in datasets:
        if not row["main_fit_equals_fit"]:
            ctx.log(f"[E5] note: the main fit of bootstrap {rep(row['dataset'])} trunk{row['trunk_bins']} "
                    f"differs from the genome's own fit by {row['main_fit_max_abs_diff']:.3g}")
    for key in TABLES:
        write_csv(lay.table(key), tables[key])
    return {key: lay.table(key) for key in TABLES}


__all__ = [
    "Layout", "NAME", "TABLES", "analyse", "band_stats", "calibration", "jobs", "report",
    "sampling_stats", "trunk_count",
]
