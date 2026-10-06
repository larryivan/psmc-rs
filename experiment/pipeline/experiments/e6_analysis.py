"""The analyses of E6 (S1-S5), as functions from parsed outputs to table rows.

Nothing here reads a path from the configuration or writes a file:
:mod:`.e6` finds the inputs, calls these functions and writes the tables.
Every function returns plain dictionaries (one per table row) so that the
tables, the tests and the figures see the same numbers.

Conventions shared with E1 (:mod:`.results`): iterations count completed EM
iterations; C psmc's log-likelihood after ``i`` iterations is printed in round
``i + 1`` (``LK``), so C psmc never evaluates its final model, while
psmc-rs's ``loglike[i]`` is the log-likelihood after ``i``. "The last common
iteration" is therefore ``n_iter - 1``.
"""

from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import demography, metrics, results

TOOLS = ("c", "rs")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """``(a - b) / |b|``, or None when either is missing or ``b`` is 0."""
    if a is None or b is None or b == 0:
        return None
    return (a - b) / abs(b)


def _max_abs(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [abs(x) for x in xs if x is not None and not math.isnan(x)]
    return max(vals) if vals else None


def _median(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(vals) if vals else None


def codes(qc: Optional[Mapping], key: str) -> str:
    """The finding codes of a psmc-rs ``qc`` block (``warnings`` or ``notes``), ``;``-joined."""
    return ";".join((qc or {}).get(key, []) or [])


# ---------------------------------------------------------------------------
# S1. EM convergence
# ---------------------------------------------------------------------------


def trace_rows(
    model: str, replicate: int, c: results.PsmcFile, rs_params: Mapping, rs: results.PsmcFile, n_iter: int
) -> List[Dict]:
    """One row per iteration ``0..n_iter`` of one replicate, both tools side by side.

    ``gain_*`` is the log-likelihood change from the previous iteration and
    ``gap_*`` the distance to the tool's own log-likelihood at the last common
    iteration (``n_iter - 1``), so both tools are measured against the same
    iteration. Parameters (theta, rho, t_max) are those after ``i`` iterations.
    """
    ll = {"c": c.loglikes(), "rs": dict(enumerate(results.rs_loglikes(rs_params)))}
    rounds = {"c": {r.round: r for r in c.complete_rounds()}, "rs": {r.round: r for r in rs.complete_rounds()}}
    last = n_iter - 1
    rows = []
    for i in range(n_iter + 1):
        row: Dict = {"model": model, "replicate": replicate, "iteration": i}
        for tool in TOOLS:
            cur = ll[tool].get(i)
            prev = ll[tool].get(i - 1)
            ref = ll[tool].get(last)
            rd = rounds[tool].get(i)
            row[f"loglike_{tool}"] = cur
            row[f"gain_{tool}"] = (cur - prev) if (cur is not None and prev is not None) else None
            row[f"gap_{tool}"] = (ref - cur) if (cur is not None and ref is not None and i <= last) else None
            row[f"theta_{tool}"] = rd.theta if rd else None
            row[f"rho_{tool}"] = rd.rho if rd else None
            row[f"t_max_{tool}"] = rd.t_max if rd else None
        row["loglike_rel_diff"] = rel_diff(row["loglike_rs"], row["loglike_c"])
        row["theta_rel_diff"] = rel_diff(row["theta_rs"], row["theta_c"])
        row["rho_rel_diff"] = rel_diff(row["rho_rs"], row["rho_c"])
        row["t_max_rel_diff"] = rel_diff(row["t_max_rs"], row["t_max_c"])
        rows.append(row)
    return rows


def _by_replicate(rows: Sequence[Mapping]) -> Dict[int, List[Mapping]]:
    out: Dict[int, List[Mapping]] = OrderedDict()
    for r in rows:
        out.setdefault(int(r["replicate"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["iteration"]))
    return out


def convergence_summary(
    model: str, tool: str, rows: Sequence[Mapping], n_iter: int, within: float, tolerance: float
) -> Dict:
    """How one tool's EM converged on one model, over its replicates (rows of :func:`trace_rows`).

    * ``replicates_monotone``: replicates whose log-likelihood never drops by
      more than ``tolerance`` from one iteration to the next;
    * ``max_decrease``: the largest drop seen (0 when there is none);
    * ``gain_last_common``: the gain of the last common iteration
      (``n_iter - 1`` over ``n_iter - 2``), a measure of how far from a
      fixed point EM stopped;
    * ``iterations_to_within``: the first iteration whose log-likelihood is
      within ``within`` units of the last common one;
    * ``theta_last_rel_change`` and ``rho_last_rel_change``: the relative
      change of the parameters in the last iteration.
    """
    reps = _by_replicate(rows)
    monotone, decreases, gains, to_within, total, theta_ch, rho_ch = 0, [], [], [], [], [], []
    for rep_rows in reps.values():
        g = [r[f"gain_{tool}"] for r in rep_rows if r[f"gain_{tool}"] is not None]
        worst = max([-x for x in g] + [0.0])
        decreases.append(worst)
        monotone += worst <= tolerance
        by_i = {int(r["iteration"]): r for r in rep_rows}
        if n_iter >= 2 and by_i[n_iter - 1][f"gain_{tool}"] is not None:
            gains.append(by_i[n_iter - 1][f"gain_{tool}"])
        first = next((int(r["iteration"]) for r in rep_rows if r[f"gap_{tool}"] is not None and r[f"gap_{tool}"] <= within), None)
        to_within.append(first)
        lo, hi = by_i[0][f"loglike_{tool}"], by_i[n_iter - 1][f"loglike_{tool}"]
        total.append(hi - lo if (lo is not None and hi is not None) else None)
        for key, acc in (("theta", theta_ch), ("rho", rho_ch)):
            a, b = by_i[n_iter][f"{key}_{tool}"], by_i[n_iter - 1][f"{key}_{tool}"]
            acc.append(abs(a / b - 1.0) if (a is not None and b) else None)
    within_known = [x for x in to_within if x is not None]
    return {
        "model": model,
        "tool": tool,
        "replicates": len(reps),
        "iterations": n_iter,
        "last_evaluated_iteration": max(
            (int(r["iteration"]) for r in rows if r[f"loglike_{tool}"] is not None), default=None
        ),
        "replicates_monotone": monotone,
        "max_decrease": max(decreases) if decreases else None,
        "gain_last_common_median": _median(gains),
        "gain_last_common_max": max(gains) if gains else None,
        "loglike_gain_total_median": _median(total),
        "iterations_to_within_median": _median(within_known),
        "iterations_to_within_max": max(within_known) if within_known else None,
        "replicates_not_within": sum(1 for x in to_within if x is None),
        "within_loglike": within,
        "theta_last_rel_change_median": _median(theta_ch),
        "rho_last_rel_change_median": _median(rho_ch),
    }


def agreement_summary(model: str, rows: Sequence[Mapping], n_iter: int) -> Dict:
    """How closely psmc-rs followed C psmc, iteration by iteration, over the replicates of a model.

    Log-likelihoods are compared as printed (C psmc: 6 decimals); theta, rho
    and t_max per round come from the ``TR`` and ``MT`` lines of both tools'
    ``.psmc`` output (6 decimals), so differences below about 1e-5 relative
    are rounding.
    """
    common = [r for r in rows if r["loglike_c"] is not None and r["loglike_rs"] is not None]
    at = lambda i: [r["loglike_rel_diff"] for r in common if int(r["iteration"]) == i]  # noqa: E731
    fitted = [r for r in rows if int(r["iteration"]) >= 1]
    final = [r for r in rows if int(r["iteration"]) == n_iter]
    return {
        "model": model,
        "replicates": len({int(r["replicate"]) for r in rows}),
        "iterations_compared": len({int(r["iteration"]) for r in common}),
        "loglike_rel_diff_iteration0_max": _max_abs(at(0)),
        "loglike_rel_diff_last_common_max": _max_abs(at(n_iter - 1)),
        "loglike_rel_diff_max": _max_abs(r["loglike_rel_diff"] for r in common),
        "theta_rel_diff_max": _max_abs(r["theta_rel_diff"] for r in fitted),
        "rho_rel_diff_max": _max_abs(r["rho_rel_diff"] for r in fitted),
        "t_max_rel_diff_max": _max_abs(r["t_max_rel_diff"] for r in fitted),
        "theta_rel_diff_final_max": _max_abs(r["theta_rel_diff"] for r in final),
    }


# ---------------------------------------------------------------------------
# S2. Format consistency
# ---------------------------------------------------------------------------


def read_psmcfa(path: Path) -> "OrderedDict[str, str]":
    """The records of a psmcfa file, ``{name: bins}`` in file order (name: header up to the first space)."""
    out: "OrderedDict[str, str]" = OrderedDict()
    name: Optional[str] = None
    parts: List[str] = []
    with open(path, encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(parts)
                words = line[1:].split()
                name, parts = (words[0] if words else ""), []
            elif line:
                parts.append(line)
    if name is not None:
        out[name] = "".join(parts)
    return out


def last_het_bin(bins: str) -> int:
    """Index of the last ``K`` bin, -1 when there is none."""
    return bins.rfind("K")


def compare_bins(model: str, fmt: str, ref: Mapping[str, str], other: Mapping[str, str]) -> List[Dict]:
    """Compare the bins of ``other`` (one format) with ``ref`` (the simulated psmcfa), per sequence.

    ``bins_differing`` counts differing bins over the common length;
    ``bins_missing_at_end`` the bins ``ref`` has beyond the end of ``other``.
    For a multihetsep input, which ends at its last row, those must be the
    bins after the last heterozygous bin of ``ref``
    (``tail_is_after_last_het``). A last row, sequence ``all``, sums them.
    """
    rows: List[Dict] = []
    names = list(ref) + [n for n in other if n not in ref]
    for name in names:
        a, b = ref.get(name, ""), other.get(name, "")
        n = min(len(a), len(b))
        av = np.frombuffer(a[:n].encode("ascii"), dtype=np.uint8)
        bv = np.frombuffer(b[:n].encode("ascii"), dtype=np.uint8)
        after_last = len(a) - (last_het_bin(a) + 1)
        missing_end = max(len(a) - len(b), 0)
        rows.append(
            {
                "model": model,
                "format": fmt,
                "sequence": name,
                "bins_psmcfa": len(a),
                "bins_format": len(b),
                "het_bins_psmcfa": a.count("K"),
                "het_bins_format": b.count("K"),
                "bins_compared": n,
                "bins_differing": int(np.count_nonzero(av != bv)),
                "bins_missing_at_end": missing_end,
                "bins_extra_at_end": max(len(b) - len(a), 0),
                "bins_after_last_het": after_last,
                "tail_is_after_last_het": missing_end in (0, after_last),
                "identical": a == b,
            }
        )
    total = {"model": model, "format": fmt, "sequence": "all"}
    for key in ("bins_psmcfa", "bins_format", "het_bins_psmcfa", "het_bins_format", "bins_compared",
                "bins_differing", "bins_missing_at_end", "bins_extra_at_end", "bins_after_last_het"):
        total[key] = sum(r[key] for r in rows)
    total["tail_is_after_last_het"] = all(r["tail_is_after_last_het"] for r in rows)
    total["identical"] = all(r["identical"] for r in rows)
    return rows + [total]


@dataclass
class FormatFit:
    """One psmc-rs fit of S2: its JSON parameters and the model they define."""

    fmt: str
    params: Mapping

    @property
    def fit(self) -> results.Fit:
        return results.fit_from_rs(self.params)


def fit_rows(
    model: str,
    fits: Mapping[str, FormatFit],
    epochs: Sequence[demography.Epoch],
    window,
    mu: float,
    gen_years: float,
    bin_size: int,
    reference: str = "psmcfa",
) -> List[Dict]:
    """One row per format: the fit, and its differences from the fit of ``reference``.

    The fits have the same pattern, so their curves are compared interval
    by interval over the reference's intervals that overlap the window
    (:func:`..metrics.interval_diff`), not at the window's times: their
    boundaries differ in years only through theta and t_max, and a time
    between two nearly equal boundaries would read a whole step.
    """
    ref = fits[reference]
    ref_curve = ref.fit.curve(mu, gen_years, bin_size)
    ref_ll = results.rs_loglikes(ref.params)
    rows = []
    for fmt, ff in fits.items():
        fit = ff.fit
        curve = fit.curve(mu, gen_years, bin_size)
        ll = results.rs_loglikes(ff.params)
        inp = ff.params.get("input", {}) or {}
        same_params = all(ff.params.get(k) == ref.params.get(k) for k in ("theta", "rho", "t_max", "lam"))
        lam_diff = _max_abs(
            rel_diff(a, b) for a, b in zip(fit.free_lambdas, ref.fit.free_lambdas)
        ) if len(fit.free_lambdas) == len(ref.fit.free_lambdas) else None
        diff = metrics.interval_diff(curve, ref_curve, (window[0], window[-1]))
        rows.append(
            {
                "model": model,
                "format": fmt,
                "input_format": inp.get("format"),
                "sequences": inp.get("sequences"),
                "bins": inp.get("bins"),
                "het_bins": inp.get("het_bins"),
                "missing_bins": inp.get("missing_bins"),
                "n_iter": ff.params.get("n_iter"),
                "theta": fit.theta,
                "rho": fit.rho,
                "t_max": fit.t_max,
                "loglike_final": ll[-1] if ll else None,
                "theta_rel_diff": rel_diff(fit.theta, ref.fit.theta),
                "rho_rel_diff": rel_diff(fit.rho, ref.fit.rho),
                "t_max_rel_diff": rel_diff(fit.t_max, ref.fit.t_max),
                "lambda_max_abs_rel_diff": lam_diff,
                "loglike_final_diff": (ll[-1] - ref_ll[-1]) if (ll and ref_ll) else None,
                "loglike_max_abs_diff": _max_abs(a - b for a, b in zip(ll, ref_ll)) if len(ll) == len(ref_ll) else None,
                "parameters_identical": same_params,
                "loglikes_identical": ll == ref_ll,
                "rmse_log10_ne_vs_psmcfa_per_interval": diff.rmse,
                "max_abs_log10_ne_vs_psmcfa_per_interval": diff.max_abs,
                "rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(curve, epochs, window, gen_years),
                "window_lo_years": float(window[0]),
                "window_hi_years": float(window[-1]),
                "qc_warnings": codes(ff.params.get("qc"), "warnings"),
                "qc_notes": codes(ff.params.get("qc"), "notes"),
            }
        )
    return rows


def curve_rows(model: str, label: str, fit: results.Fit, mu: float, gen_years: float, bin_size: int) -> List[Dict]:
    """The step curve of a fit: one row per interval (left edge in years, Ne)."""
    curve = fit.curve(mu, gen_years, bin_size)
    return [
        {"model": model, "format": label, "interval": k, "years": y, "ne": ne}
        for k, (y, ne) in enumerate(zip(curve.years, curve.ne))
    ]


def truth_rows(model: str, epochs: Sequence[demography.Epoch], years, gen_years: float) -> List[Dict]:
    """The true Ne(t) at ``years`` (format ``truth``)."""
    ne = metrics.truth_on_grid(epochs, years, gen_years)
    return [{"model": model, "format": "truth", "interval": i, "years": float(y), "ne": float(n)}
            for i, (y, n) in enumerate(zip(years, ne))]


# ---------------------------------------------------------------------------
# S3. TMRCA decoding
# ---------------------------------------------------------------------------

POSTERIOR_COLUMNS = (
    "seq_id", "seq_bin", "obs", "map_state", "tmrca_map_years", "tmrca_mean_years",
    "tmrca_q025_years", "tmrca_q975_years",
)


def bin_tmrca(intervals: Sequence[Tuple[int, int, float]], length: int, bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """The true TMRCA of each bin of a sequence, and how many intervals each bin overlaps.

    ``intervals`` are the run-length TMRCA intervals of the sequence
    (``start``, ``end`` exclusive, TMRCA), covering ``[0, length)`` without
    gaps. A bin's TMRCA is the length-weighted mean over its bases (most bins
    lie within one interval: the TMRCA changes every few kb). Bins are those
    of fq2psmcfa: ``ceil(length / bin_size)``, the last one possibly partial.
    """
    if not intervals:
        raise ValueError("no TMRCA intervals")
    starts = np.array([s for s, _, _ in intervals], dtype=np.int64)
    ends = np.array([e for _, e, _ in intervals], dtype=np.int64)
    tm = np.array([t for _, _, t in intervals], dtype=float)
    if starts[0] != 0 or ends[-1] != length or np.any(starts[1:] != ends[:-1]):
        raise ValueError(f"TMRCA intervals do not tile [0, {length})")
    cum = np.concatenate([[0.0], np.cumsum((ends - starts) * tm)])
    knots = np.concatenate([starts, ends[-1:]]).astype(float)
    n_bins = (length + bin_size - 1) // bin_size
    edges = np.minimum(np.arange(n_bins + 1, dtype=np.int64) * bin_size, length)
    integral = np.interp(edges.astype(float), knots, cum)
    mean = np.diff(integral) / np.diff(edges)
    first = np.searchsorted(starts, edges[:-1], side="right") - 1
    last = np.searchsorted(starts, edges[1:] - 1, side="right") - 1
    return mean, (last - first + 1)


def state_of(years, left_edges) -> np.ndarray:
    """The PSMC state (0-based interval) holding each time of ``years``; ``left_edges`` from :meth:`.results.Fit.curve`."""
    return np.searchsorted(np.asarray(left_edges, dtype=float), np.asarray(years, dtype=float), side="right") - 1


@dataclass
class Decoded:
    """Per-bin truth and posterior summaries of S3, aligned bin by bin (all times in years)."""

    true: np.ndarray
    true_changes: np.ndarray  # intervals overlapping each bin
    mean: np.ndarray
    map: np.ndarray
    q025: np.ndarray
    q975: np.ndarray
    obs: np.ndarray  # "T", "K" or "N"
    map_state: np.ndarray
    sequence: np.ndarray  # 0-based sequence index
    seq_bin: np.ndarray
    left_edges: np.ndarray  # PSMC interval boundaries in years (the last interval is open)

    def states(self, years) -> np.ndarray:
        return state_of(years, self.left_edges)

    def covered_point(self) -> np.ndarray:
        """Truth between the 2.5% and 97.5% posterior quantiles (each at its state's average time)."""
        return (self.true >= self.q025) & (self.true <= self.q975)

    def covered_states(self) -> np.ndarray:
        """Truth inside the time intervals of the states from the 2.5% to the 97.5% quantile."""
        edges = np.append(self.left_edges, np.inf)
        lo = edges[self.states(self.q025)]
        hi = edges[self.states(self.q975) + 1]
        return (self.true >= lo) & (self.true < hi)


def align(posterior, truths: Sequence[Tuple[np.ndarray, np.ndarray]], left_edges, gen_years: float) -> Decoded:
    """Pair the posterior rows (a DataFrame of :data:`POSTERIOR_COLUMNS`) with the true TMRCA of each bin.

    ``truths[k]`` is :func:`bin_tmrca` of sequence ``k`` (input order, as
    psmc-rs numbers them), in generations; they are scaled to years with
    ``gen_years``. Every sequence must have as many posterior rows as bins.
    """
    seq = posterior["seq_id"].to_numpy()
    sbin = posterior["seq_bin"].to_numpy()
    true = np.empty(len(posterior))
    changes = np.empty(len(posterior), dtype=np.int64)
    counts = np.bincount(seq, minlength=len(truths))
    if len(counts) != len(truths):
        raise ValueError(f"the posterior has {len(counts)} sequences, the truth {len(truths)}")
    for k, (t, n) in enumerate(truths):
        if counts[k] != len(t):
            raise ValueError(f"sequence {k}: {counts[k]} posterior bins, {len(t)} true bins")
        idx = np.flatnonzero(seq == k)
        true[idx] = t[sbin[idx]] * gen_years
        changes[idx] = n[sbin[idx]]
    return Decoded(
        true=true,
        true_changes=changes,
        mean=posterior["tmrca_mean_years"].to_numpy(dtype=float),
        map=posterior["tmrca_map_years"].to_numpy(dtype=float),
        q025=posterior["tmrca_q025_years"].to_numpy(dtype=float),
        q975=posterior["tmrca_q975_years"].to_numpy(dtype=float),
        obs=posterior["obs"].to_numpy(dtype=str),
        map_state=posterior["map_state"].to_numpy(),
        sequence=seq,
        seq_bin=sbin,
        left_edges=np.asarray(left_edges, dtype=float),
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    from scipy import stats

    if len(x) < 2:
        return None
    return float(stats.spearmanr(x, y).statistic)


def state_mismatches(d: Decoded) -> int:
    """Bins whose MAP time does not fall in the interval of their MAP state.

    psmc-rs places each state at its average coalescent time, which lies in
    the state's interval; a mismatch would mean that the interval boundaries
    (``left_edges``, from the fit's theta, t_max and the scaling) are not the
    ones psmc-rs used, and the interval coverage would be wrong.
    """
    return int(np.count_nonzero(d.states(d.map) != d.map_state))


def tmrca_summary(d: Decoded, label: str, mask: Optional[np.ndarray] = None) -> Dict:
    """Agreement of the posterior with the truth over the bins in ``mask`` (all by default)."""
    m = np.ones(len(d.true), dtype=bool) if mask is None else mask
    lt, lmean, lmap = np.log10(d.true[m]), np.log10(d.mean[m]), np.log10(d.map[m])
    err = lmean - lt
    return {
        "bins": label,
        "n_bins": int(m.sum()),
        "share_bins_with_tmrca_change": float(np.mean(d.true_changes[m] > 1)) if m.any() else None,
        "pearson_log10_mean": _pearson(lt, lmean),
        "spearman_mean": _spearman(lt, lmean),
        "pearson_log10_map": _pearson(lt, lmap),
        "spearman_map": _spearman(lt, lmap),
        "rmse_log10_mean": float(np.sqrt(np.mean(err**2))) if m.any() else None,
        "mean_log10_error_mean": float(np.mean(err)) if m.any() else None,
        "median_abs_log10_error_mean": float(np.median(np.abs(err))) if m.any() else None,
        "rmse_log10_map": float(np.sqrt(np.mean((lmap - lt) ** 2))) if m.any() else None,
        "map_state_is_true_state": float(np.mean(d.states(d.map[m]) == d.states(d.true[m]))) if m.any() else None,
        "coverage95_point": float(np.mean(d.covered_point()[m])) if m.any() else None,
        "coverage95_states": float(np.mean(d.covered_states()[m])) if m.any() else None,
        "median_log10_width95": float(np.median(np.log10(d.q975[m] / d.q025[m]))) if m.any() else None,
        "median_true_years": float(np.median(d.true[m])) if m.any() else None,
        "median_posterior_mean_years": float(np.median(d.mean[m])) if m.any() else None,
    }


def calibration_rows(d: Decoded, by: str, classes: int) -> List[Dict]:
    """Coverage and error in ``classes`` quantile classes of the true TMRCA (``by="true"``) or the posterior mean."""
    key = {"true": d.true, "posterior_mean": d.mean}[by]
    edges = np.quantile(key, np.linspace(0.0, 1.0, classes + 1))
    cls = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, classes - 1)
    point, states = d.covered_point(), d.covered_states()
    lt, lm = np.log10(d.true), np.log10(d.mean)
    rows = []
    for k in range(classes):
        m = cls == k
        if not m.any():
            continue
        rows.append(
            {
                "by": by,
                "class": k + 1,
                "lo_years": float(edges[k]),
                "hi_years": float(edges[k + 1]),
                "n_bins": int(m.sum()),
                "median_true_years": float(np.median(d.true[m])),
                "median_posterior_mean_years": float(np.median(d.mean[m])),
                "median_map_years": float(np.median(d.map[m])),
                "mean_log10_error_mean": float(np.mean(lm[m] - lt[m])),
                "coverage95_point": float(np.mean(point[m])),
                "coverage95_states": float(np.mean(states[m])),
                "median_log10_width95": float(np.median(np.log10(d.q975[m] / d.q025[m]))),
            }
        )
    return rows


def density_rows(d: Decoded, cells: int, coverage: float = 0.999) -> List[Dict]:
    """Counts of bins on a ``cells`` x ``cells`` grid of log10 true TMRCA x log10 posterior mean (non-empty cells).

    The grid spans the central ``coverage`` share of both variables
    together; bins outside it are counted in the edge cells.
    """
    lt, lm = np.log10(d.true), np.log10(d.mean)
    tail = (1.0 - coverage) / 2.0
    lo = min(np.quantile(lt, tail), np.quantile(lm, tail))
    hi = max(np.quantile(lt, 1 - tail), np.quantile(lm, 1 - tail))
    edges = np.linspace(lo, hi, cells + 1)
    counts, _, _ = np.histogram2d(np.clip(lt, lo, hi), np.clip(lm, lo, hi), bins=[edges, edges])
    rows = []
    for i, j in zip(*np.nonzero(counts)):
        rows.append(
            {
                "log10_true_lo": float(edges[i]), "log10_true_hi": float(edges[i + 1]),
                "log10_mean_lo": float(edges[j]), "log10_mean_hi": float(edges[j + 1]),
                "n_bins": int(counts[i, j]),
            }
        )
    return rows


def track_rows(d: Decoded, sequence: int, name: str, start_bin: int, n_bins: int, bin_size: int) -> List[Dict]:
    """The bins ``start_bin .. start_bin + n_bins`` of one sequence (the example track)."""
    m = (d.sequence == sequence) & (d.seq_bin >= start_bin) & (d.seq_bin < start_bin + n_bins)
    idx = np.flatnonzero(m)
    idx = idx[np.argsort(d.seq_bin[idx], kind="stable")]
    return [
        {
            "sequence": name,
            "start_bp": int(d.seq_bin[i]) * bin_size,
            "obs": str(d.obs[i]),
            "true_years": float(d.true[i]),
            "posterior_mean_years": float(d.mean[i]),
            "map_years": float(d.map[i]),
            "q025_years": float(d.q025[i]),
            "q975_years": float(d.q975[i]),
        }
        for i in idx
    ]


# ---------------------------------------------------------------------------
# S5. QC of the real genomes
# ---------------------------------------------------------------------------


def qc_row(sample_id: str, label: str, qc: Mapping) -> Dict:
    """The key metrics and finding codes of one ``psmc-rs qc --json`` report."""
    t = qc.get("totals", {}) or {}
    lengths = qc.get("lengths", {}) or {}
    below = lengths.get("below_trunk") or {}
    sat = qc.get("saturation", {}) or {}
    findings = qc.get("findings", []) or []
    return {
        "sample_id": sample_id,
        "label": label,
        "input_format": qc.get("input_format"),
        "bin_size": qc.get("bin_size"),
        "sequences": qc.get("n_sequences"),
        "bins": t.get("bins"),
        "missing_fraction": t.get("missing_fraction"),
        "het_bin_fraction": t.get("het_bin_fraction"),
        "theta0_per_bin": t.get("theta0_per_bin"),
        "het_per_kb": t.get("het_per_kb"),
        "called_bases": t.get("called_bases_from_bins"),
        "n50_bases": lengths.get("n50_bases"),
        "longest_bases": lengths.get("longest_bases"),
        "longest_missing_run_bins": t.get("longest_missing_run_bins"),
        "multi_het_share": sat.get("multi_het_share"),
        "max_bin_size_below_target": sat.get("max_bin_size_below_target"),
        "bootstrap_trunk_bins": qc.get("bootstrap_trunk_bins"),
        "bootstrap_trunks": lengths.get("bootstrap_trunks"),
        "share_bins_below_trunk": below.get("fraction"),
        "expected_recombinations": qc.get("expected_recombinations"),
        "warnings": ";".join(f["code"] for f in findings if f.get("severity") == "warning"),
        "notes": ";".join(f["code"] for f in findings if f.get("severity") != "warning"),
    }


def finding_rows(sample_id: str, qc: Mapping) -> List[Dict]:
    """One row per finding of a QC report: code, severity, message, recommendation."""
    return [
        {
            "sample_id": sample_id,
            "code": f.get("code"),
            "severity": f.get("severity"),
            "message": f.get("message"),
            "recommendation": f.get("recommendation"),
            "sequences_named": len(f.get("sequences") or []),
        }
        for f in qc.get("findings", []) or []
    ]


__all__ = [
    "Decoded", "FormatFit", "agreement_summary", "align", "bin_tmrca", "calibration_rows", "compare_bins",
    "convergence_summary", "curve_rows", "density_rows", "finding_rows", "fit_rows",
    "last_het_bin", "qc_row", "read_psmcfa", "state_mismatches", "state_of",
    "tmrca_summary", "trace_rows", "track_rows", "truth_rows",
]
