"""Small stand-in tables in the schemas of :mod:`.schemas`, for developing and testing the figures.

They are made from the true Ne(t) of the demographic models with seeded
noise, sized by a profile (``tiny`` for the unit tests, ``smoke``, or
``production`` to see a figure at its real size); none of their numbers is
a result. They are written to a scratch runs directory, never to
``experiment/runs``::

    python -m experiment.pipeline.figures.fake --runs-dir /tmp/fake [--profile smoke] [E2 E3 ...]
    python -m experiment.pipeline.figures --runs-dir /tmp/fake --profile smoke

E5's calibration table is computed from the stand-in curves by
:func:`calibration`, which is the definition the figure expects of
``E5_calibration.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, paths
from ..results import boundaries
from .schemas import SCHEMAS

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")
N_STEPS = 63  # intervals - 1 of the 4+25*2+4+6 pattern
T_MAX = 13.0  # a typical fitted t_max (2*N0 generations)


def _rng(*key) -> "object":
    import numpy as np

    digest = hashlib.sha256("/".join(map(str, ("fake",) + key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def write(runs_dir: Path, key: str, rows: Sequence[Dict]) -> Path:
    """Write ``rows`` as table ``key`` (columns: the schema's, then any others in the rows)."""
    schema = SCHEMAS[key]
    path = schema.path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(schema.names)
    for r in rows[:1]:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return path


def fake_curve(size_at_years, n0: float, gen_years: float, rng, sd: float = 0.08,
               offsets: Optional[Sequence[float]] = None) -> List[Tuple[int, float, float]]:
    """``(interval, years, ne)`` of a PSMC-like step curve around ``size_at_years``.

    The intervals are C psmc's for t_max 13 and 64 intervals with N0
    ``n0``; each interval's Ne is the truth at its geometric middle times
    a log-normal error of SD ``sd`` (log10), plus ``offsets[k]`` (log10).
    """
    t = boundaries(T_MAX, N_STEPS)
    years = [2.0 * n0 * x * gen_years for x in t]
    out = []
    for k in range(N_STEPS + 1):
        lo = years[k] if k else years[1] / 2.0
        hi = years[k + 1] if k < N_STEPS else years[k] * 1.5
        mid = math.sqrt(lo * hi)
        err = rng.normal(0.0, sd) + (offsets[k] if offsets is not None else 0.0)
        out.append((k, years[k], float(size_at_years(mid)) * 10 ** err))
    return out


def model_size(model: str, gen_years: float, scale: float = 1.0, time_scale: float = 1.0):
    eps = demography.epochs(config.MODELS[model])
    return lambda years: scale * demography.ne_at(eps, years / gen_years / time_scale)


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------


def e1(runs_dir: Path, profile: config.Profile) -> List[Path]:
    import numpy as np

    cfg = profile.e1
    g = cfg.scaling.gen_years
    curves, reps, pairs, traces, truth, table1 = [], [], [], [], [], []
    dense = np.logspace(2, 8, 400)
    for m in cfg.models:
        eps = demography.epochs(config.MODELS[m])
        lo, hi = metrics.informative_window(eps, cfg.grid.quantiles, g)
        table1.append({"model": m, "replicates": cfg.replicates, "window_lo_years": lo, "window_hi_years": hi})
        truth += [{"model": m, "years": float(t), "ne": demography.ne_at(eps, t / g)} for t in dense]
        n0 = config.MODELS[m].initial_size
        for r in range(1, cfg.replicates + 1):
            rng = _rng("E1", m, r)
            base = fake_curve(model_size(m, g), n0, g, rng, sd=0.1)
            for tool in ("c", "rs"):
                trng = _rng("E1", m, r, tool)
                for k, y, ne in base:
                    v = ne * 10 ** (trng.normal(0, 1e-3) if tool == "rs" else 0.0)
                    curves.append({"model": m, "replicate": r, "tool": tool, "interval": k, "years": y, "ne": v,
                                   "t_2n0": y / (2 * n0 * g), "lambda": v / n0})
                reps.append({"model": m, "replicate": r, "tool": tool,
                             "rmse_log10_ne_vs_truth": 0.12 + 0.03 * rng.random() + (0.002 if tool == "rs" else 0)})
                ll0 = -1.0e5 - 2e3 * rng.random()
                for i in range(cfg.inference.n_iter + 1):
                    ll = ll0 + 900.0 * (1 - math.exp(-0.6 * i)) + (trng.normal(0, 0.01) if tool == "rs" else 0.0)
                    traces.append({"model": m, "replicate": r, "tool": tool, "iteration": i,
                                   "loglike": None if (tool == "c" and i == cfg.inference.n_iter) else ll,
                                   "theta": 1e-3 * (1 + 0.05 * (1 - math.exp(-0.5 * i))),
                                   "rho": 2e-4 * (1 + 0.1 * (1 - math.exp(-0.4 * i)))})
            pairs.append({"model": m, "replicate": r, "rmse_log10_ne_rs_vs_c": 1e-3 * (1 + rng.random()),
                          "max_abs_log10_ne_diff": 3e-3 * (1 + rng.random())})
    return [write(runs_dir, "E1_curves", curves), write(runs_dir, "E1_truth", truth),
            write(runs_dir, "E1_table1", table1), write(runs_dir, "E1_replicates", reps),
            write(runs_dir, "E1_rs_vs_c", pairs), write(runs_dir, "E1_traces", traces)]


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------

# Shapes of the stand-in histories: (model, size scale, time scale).
_E2_SHAPES = {"HLemySub1": ("expansion", 6.0, 3.0), "HLhydTec1": ("bottleneck", 3.0, 2.0),
              "HLpelCas1": ("zigzag", 1.2, 8.0), "Papuan": ("zigzag", 0.3, 1.0)}


def e2(runs_dir: Path, profile: config.Profile) -> List[Path]:
    cfg = profile.e2
    curves, summary = [], []
    for gnm in cfg.genomes:
        model, scale, tscale = _E2_SHAPES.get(gnm.sample_id, ("constant", 1.0, 1.0))
        g = gnm.scaling.gen_years
        size = model_size(model, g, scale, tscale)
        n0 = config.MODELS[model].initial_size * scale
        base = fake_curve(size, n0, g * tscale, _rng("E2", gnm.sample_id), sd=0.05)
        for tool in ("c", "rs"):
            rng = _rng("E2", gnm.sample_id, tool)
            for k, y, ne in base:
                curves.append({"sample": gnm.sample_id, "tool": tool, "replicate": 0, "interval": k, "years": y,
                               "ne": ne * 10 ** (rng.normal(0, 2e-3) if tool == "rs" else 0.0)})
        for b in range(1, cfg.bootstrap_replicates + 1):
            rng = _rng("E2", gnm.sample_id, "boot", b)
            shift = rng.normal(0, 0.01)
            for k, y, ne in base:
                edge = 0.25 if (k < 4 or k > N_STEPS - 6) else 0.06
                curves.append({"sample": gnm.sample_id, "tool": "rs", "replicate": b, "interval": k,
                               "years": y * 10 ** shift, "ne": ne * 10 ** rng.normal(0, edge)})
        summary.append({"sample": gnm.sample_id, "mu": gnm.scaling.mu, "gen_years": g,
                        "rmse_log10_ne_rs_vs_c": 2e-3, "window_lo_years": base[2][1],
                        "window_hi_years": base[-3][1], "bootstrap_replicates": cfg.bootstrap_replicates})
    return [write(runs_dir, "E2_curves", curves), write(runs_dir, "E2_summary", summary)]


# ---------------------------------------------------------------------------
# E3
# ---------------------------------------------------------------------------


def e3(runs_dir: Path, profile: config.Profile) -> List[Path]:
    cfg, costs = profile.e3, profile.costs
    n = cfg.inference.n_iter
    runs, threads = [], []

    def speedup(t: int) -> float:
        return t ** 0.85

    for size in cfg.genome_sizes_mb:
        c_gb = costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * size
        series = [("c", 1, costs.c_psmc_s_per_mb_iter * size * n, c_gb * 1024, c_gb > cfg.memory_limit_gb),
                  ("rs01", 1, 4.0 * costs.rs02_s_per_mb_iter * size * n, 80.0 + 0.6 * size, False)]
        series += [("rs02", t, costs.rs02_s_per_mb_iter * size * n / speedup(t), 12.0 + 0.012 * size + 2.0 * t, False)
                   for t in cfg.rs02_threads]
        for tool, t, wall, rss, skip in series:
            for rep in range(1, cfg.repeats + 1):
                rng = _rng("E3", tool, t, size, rep)
                runs.append({"tool": tool, "threads": t, "genome_mb": size, "repeat": rep,
                             "status": "skipped" if skip else "ok",
                             "wall_seconds": None if skip else wall * (1 + rng.normal(0, 0.03)),
                             "peak_rss_mb": None if skip else rss * (1 + rng.normal(0, 0.01)),
                             "note": f"estimated {c_gb:.1f} GB > {cfg.memory_limit_gb:g} GB" if skip else ""})
    base = costs.rs02_s_per_mb_iter * cfg.thread_scaling_size_mb * n
    for t in cfg.thread_scaling_threads:
        for rep in range(1, cfg.repeats + 1):
            rng = _rng("E3", "threads", t, rep)
            threads.append({"genome_mb": cfg.thread_scaling_size_mb, "threads": t, "repeat": rep,
                            "wall_seconds": base / speedup(t) * (1 + rng.normal(0, 0.02)),
                            "peak_rss_mb": 12.0 + 0.012 * cfg.thread_scaling_size_mb + 2.0 * t,
                            "json_sha256": hashlib.sha256(b"fake fit").hexdigest()})
    return [write(runs_dir, "E3_runs", runs), write(runs_dir, "E3_threads", threads)]


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------


def e4(runs_dir: Path, profile: config.Profile) -> List[Path]:
    cfg = profile.e4
    g = cfg.scaling.gen_years
    masks = [("none", "none", 0.0)] + [(m.name, m.structure, m.missing_fraction) for m in cfg.masks]
    size = model_size(cfg.model, g)
    n0 = config.MODELS[cfg.model].initial_size
    rows, curves = [], []
    for r in range(1, cfg.replicates + 1):
        base = fake_curve(size, n0, g, _rng("E4", r), sd=0.08)
        for mask, structure, f in masks:
            for route in cfg.routes:
                rng = _rng("E4", r, mask, route)
                if route in ("exact", "fq2psmcfa") or mask == "none":
                    bias, offs, differ = (0.0 if mask == "none" else rng.normal(0, 0.004)), 0.0, 0.0
                    rng_curve = _rng("E4", r, mask, "exact")
                elif route == "multihetsep":
                    bias = (0.06 if structure == "dispersed" else 0.002) * f / 0.3
                    offs = 0.05 * f / 0.3 if structure == "dispersed" else 0.0
                    differ = (0.4 if structure == "dispersed" else 0.004) * f
                    rng_curve = rng
                else:  # nomask
                    bias, offs, differ = -0.95 * f, math.log10(1 - 0.95 * f), f * 0.9
                    rng_curve = rng
                offsets = [offs * (1.0 if k < 20 else 0.6) for k in range(N_STEPS + 1)]
                fit = [(k, y, ne * 10 ** (rng_curve.normal(0, 0.02) + offsets[k])) for k, y, ne in base]
                for k, y, ne in fit:
                    curves.append({"replicate": r, "mask": mask, "route": route, "interval": k, "years": y, "ne": ne})
                rows.append({"replicate": r, "mask": mask, "structure": structure, "missing_fraction": f,
                             "route": route, "theta": 1e-3 * (1 + bias), "theta_rel_bias": bias,
                             "rmse_log10_ne_vs_truth": 0.12 + abs(offs) + 0.01 * rng.random(),
                             "bins_differ_fraction": differ,
                             "qc_codes": "VARIANTS_ONLY_NO_MASK" if route == "nomask" else
                             ("APPROX_MISSING_PLACEMENT" if route == "multihetsep" else "")})
    return [write(runs_dir, "E4_routes", rows), write(runs_dir, "E4_curves", curves)]


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------


def calibration(fits: Dict[int, "object"], boots: Dict[Tuple[int, int], "object"], truth, band: Tuple[float, float]
                ) -> List[Dict]:
    """The rows of ``E5_calibration.csv`` from log10 Ne on the grid.

    ``fits[dataset]`` is log10 Ne of the fit of each independent genome (a
    vector over the grid), ``boots[(dataset, trunk)]`` a replicates x grid
    array of log10 Ne of that dataset's bootstrap, ``truth`` the true log10
    Ne on the grid. The true sampling SD is the SD over ``fits``; for each
    trunk and dataset the bootstrap SD is the SD over its replicates, the
    band its ``band`` quantiles; coverage is the fraction of datasets whose
    band contains the truth.
    """
    import numpy as np

    sd_true = np.std(np.vstack(list(fits.values())), axis=0, ddof=1)
    out = []
    for trunk in sorted({t for _, t in boots}, reverse=True):
        ds = sorted(d for d, t in boots if t == trunk)
        sds = np.vstack([np.std(boots[(d, trunk)], axis=0, ddof=1) for d in ds])
        cover = np.vstack([(np.quantile(boots[(d, trunk)], band[0], axis=0) <= truth)
                           & (truth <= np.quantile(boots[(d, trunk)], band[1], axis=0)) for d in ds])
        ratio = sds / sd_true
        for i in range(len(truth)):
            out.append({"trunk_bins": trunk, "grid_index": i,
                        "sd_true": float(sd_true[i]), "sd_boot_median": float(np.median(sds[:, i])),
                        "sd_boot_q25": float(np.quantile(sds[:, i], 0.25)),
                        "sd_boot_q75": float(np.quantile(sds[:, i], 0.75)),
                        "sd_ratio_median": float(np.median(ratio[:, i])),
                        "coverage": float(np.mean(cover[:, i])), "n_datasets": len(ds)})
    return out


def e5(runs_dir: Path, profile: config.Profile) -> List[Path]:
    import numpy as np

    from ..results import Curve

    cfg = profile.e5
    g = cfg.scaling.gen_years
    size = model_size(cfg.model, g)
    n0 = config.MODELS[cfg.model].initial_size
    eps = demography.epochs(config.MODELS[cfg.model])
    grid = metrics.model_grid(eps, cfg.grid, g)
    truth = np.log10(metrics.truth_on_grid(eps, grid, g))
    curves, fits, boots = [], {}, {}
    for d in range(1, cfg.truth_replicates + 1):
        fit = fake_curve(size, n0, g, _rng("E5", d), sd=0.1)
        curves += [{"kind": "fit", "dataset": d, "trunk_bins": 0, "replicate": 0, "interval": k, "years": y, "ne": ne}
                   for k, y, ne in fit]
        c = Curve(tuple(y for _, y, _ in fit), tuple(ne for _, _, ne in fit))
        fits[d] = np.log10(c.at(grid))
        if d > cfg.bootstrap_datasets:
            continue
        for ti, trunk in enumerate(cfg.trunk_bins):
            reps = []
            for b in range(1, cfg.bootstrap_replicates + 1):
                rng = _rng("E5", d, trunk, b)
                boot = [(k, y, ne * 10 ** rng.normal(0, 0.075 + 0.015 * ti)) for k, y, ne in fit]
                curves += [{"kind": "bootstrap", "dataset": d, "trunk_bins": trunk, "replicate": b, "interval": k,
                            "years": y, "ne": ne} for k, y, ne in boot]
                reps.append(np.log10(Curve(tuple(y for _, y, _ in boot), tuple(ne for _, _, ne in boot)).at(grid)))
            boots[(d, trunk)] = np.vstack(reps)
    rows = calibration(fits, boots, truth, cfg.band)
    for row in rows:
        row["years"] = float(grid[row.pop("grid_index")])
    return [write(runs_dir, "E5_calibration", rows), write(runs_dir, "E5_curves", curves)]


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------


def e6(runs_dir: Path, profile: config.Profile) -> List[Path]:
    import numpy as np

    e1cfg, cfg = profile.e1, profile.e6
    g = e1cfg.scaling.gen_years
    bins = e1cfg.genome.length // e1cfg.inference.bin_size
    formats, fcurves = [], []
    for m in e1cfg.models:
        rng = _rng("E6", m)
        base = fake_curve(model_size(m, g), config.MODELS[m].initial_size, g, rng, sd=0.1)
        tail = int(rng.integers(0, 40))
        for inp in ("psmcfa", "vcf", "multihetsep"):
            eps = 0.0 if inp != "multihetsep" else 2e-4 * tail / 40
            formats.append({"model": m, "input": inp, "bins": bins - (tail if inp == "multihetsep" else 0),
                            "bins_differ": 0, "tail_bins_missing": tail if inp == "multihetsep" else 0,
                            "rmse_log10_ne_vs_psmcfa": eps})
            frng = _rng("E6", m, inp)
            fcurves += [{"model": m, "input": inp, "interval": k, "years": y,
                         "ne": ne * 10 ** (frng.normal(0, eps) if eps else 0.0)} for k, y, ne in base]
    # TMRCA: a true track of exponential segments, estimates around it.
    rng = _rng("E6", "tmrca")
    n_bins = min(cfg.tmrca_example_track_bp // e1cfg.inference.bin_size, 4000)
    true = np.empty(n_bins)
    i = 0
    while i < n_bins:
        length = int(rng.exponential(60)) + 1
        true[i:i + length] = rng.lognormal(math.log(4e4 * g), 1.0)
        i += length
    mean = true * np.exp(rng.normal(0, 0.45, n_bins))
    track = [{"start_bp": k * e1cfg.inference.bin_size, "true_tmrca_years": float(true[k]),
              "post_mean_years": float(mean[k]), "post_map_years": float(mean[k] * 0.8),
              "q025_years": float(mean[k] / 3.5), "q975_years": float(mean[k] * 3.5)} for k in range(n_bins)]
    big_true = rng.lognormal(math.log(4e4 * g), 1.0, 20000)
    edges = np.logspace(3, 8, 41)
    hist, calib, summary = [], [], []
    for est, f in (("mean", 1.0), ("map", 0.8)):
        big_est = big_true * f * np.exp(rng.normal(0, 0.45, big_true.size))
        h, _, _ = np.histogram2d(big_true, big_est, bins=[edges, edges])
        for a in range(len(edges) - 1):
            for b in range(len(edges) - 1):
                if h[a, b]:
                    hist.append({"estimator": est, "true_lo_years": edges[a], "true_hi_years": edges[a + 1],
                                 "est_lo_years": edges[b], "est_hi_years": edges[b + 1], "count": int(h[a, b])})
        summary.append({"estimator": est, "pearson_log10": float(np.corrcoef(np.log10(big_true), np.log10(big_est))[0, 1]),
                        "coverage": 0.93, "bins": int(big_true.size)})
    cedges = np.logspace(4, 7, 13)
    for a in range(len(cedges) - 1):
        sel = (big_true >= cedges[a]) & (big_true < cedges[a + 1])
        if sel.sum():
            calib.append({"true_lo_years": cedges[a], "true_hi_years": cedges[a + 1], "bins": int(sel.sum()),
                          "coverage": float(np.clip(0.95 - 0.08 * abs(a - 6) / 6 + rng.normal(0, 0.01), 0, 1))})
    return [write(runs_dir, "E6_formats", formats), write(runs_dir, "E6_format_curves", fcurves),
            write(runs_dir, "E6_tmrca_track", track), write(runs_dir, "E6_tmrca_hist2d", hist),
            write(runs_dir, "E6_tmrca_calibration", calib), write(runs_dir, "E6_tmrca_summary", summary)]


WRITERS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def write_all(runs_dir: Path, profile: config.Profile, experiments: Iterable[str] = EXPERIMENTS) -> List[Path]:
    """Write the stand-in tables of ``experiments`` to ``runs_dir`` (refusing experiment/runs)."""
    runs_dir = Path(runs_dir).resolve()
    if runs_dir == paths.RUNS.resolve() or runs_dir == paths.default_runs_dir("production").resolve():
        raise ValueError(f"refusing to write stand-in tables to the production runs directory {runs_dir}")
    out: List[Path] = []
    for exp in experiments:
        out += WRITERS[exp](runs_dir, profile)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.figures.fake", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", metavar="EXP",
                   help=f"experiments to write, of {' '.join(EXPERIMENTS)} (default: E2-E6, which have no tables yet)")
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--profile", default="smoke", choices=sorted(config.PROFILES))
    args = p.parse_args(argv)
    unknown = [e for e in args.experiments if e not in WRITERS]
    if unknown:
        p.error(f"unknown experiment(s) {' '.join(unknown)}; choose from {' '.join(EXPERIMENTS)}")
    for path in write_all(args.runs_dir, config.get_profile(args.profile), args.experiments or EXPERIMENTS[1:]):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
