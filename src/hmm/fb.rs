//! The per-sequence forward-backward engine shared by the E-step, the
//! log-likelihood and posterior decoding.
//!
//! Memory is bounded by checkpointing: the forward pass keeps the forward
//! vector entering every segment of [`CHECKPOINT_STRIDE`] sites, and the
//! backward pass recomputes one segment's forward vectors at a time from its
//! checkpoint. Recomputation runs the same kernel on the same inputs, so the
//! recomputed vectors are bit-identical to the ones the forward pass produced,
//! and results do not depend on whether full rows were cached instead.

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::budget::AlphaCacheBudget;
use super::kernel::Kernel;
use super::obs::{ObsTable, obs_class};
use super::{CHECKPOINT_STRIDE, Cancelled};
use crate::io::BinCounts;

const K: usize = CHECKPOINT_STRIDE;

/// The observations of one sequence: one or more rows chained into one HMM
/// chain, with the callable bases of their bins when the input has them.
pub(crate) struct SeqObs<'a> {
    rows: &'a [Vec<u8>],
    /// The counts of `rows`, and the table that maps a bin's symbol and
    /// count to its observation index.
    counted: Option<(&'a [BinCounts], &'a ObsTable)>,
    /// `offsets[r]` is the index of the first site of row `r`; the last entry is the length.
    offsets: Vec<usize>,
}

impl<'a> SeqObs<'a> {
    /// `rows` must be non-empty and contain no empty row. `counts`, when
    /// given, holds the counts of the bins of each row, and `table` indexes
    /// every pair of symbol and count they hold ([`ObsTable::counted`]).
    pub(crate) fn new(
        rows: &'a [Vec<u8>],
        counts: Option<&'a [BinCounts]>,
        table: &'a ObsTable,
    ) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut pos = 0usize;
        offsets.push(0);
        for r in rows {
            pos += r.len();
            offsets.push(pos);
        }
        Self {
            rows,
            counted: counts.map(|c| (c, table)),
            offsets,
        }
    }

    pub(crate) fn len(&self) -> usize {
        *self.offsets.last().expect("offsets is never empty")
    }

    pub(crate) fn n_segments(&self) -> usize {
        self.len().div_ceil(K)
    }

    /// First site and length of segment `s`.
    pub(crate) fn segment(&self, s: usize) -> (usize, usize) {
        let start = s * K;
        (start, K.min(self.len() - start))
    }

    /// Observation indices of sites `start..start + out.len()`: the symbol
    /// classes without counts, the indices of the observation table with
    /// them.
    pub(crate) fn read(&self, start: usize, out: &mut [u16]) {
        let mut r = self.offsets.partition_point(|&o| o <= start) - 1;
        let mut pos = start;
        let mut k = 0;
        while k < out.len() {
            let row = &self.rows[r];
            let off = pos - self.offsets[r];
            let take = (row.len() - off).min(out.len() - k);
            let symbols = &row[off..off + take];
            let dst = &mut out[k..k + take];
            match self.counted {
                None => {
                    for (d, &v) in dst.iter_mut().zip(symbols) {
                        *d = u16::from(obs_class(v));
                    }
                }
                Some((counts, table)) => {
                    counts[r].read_into(off, dst);
                    for (d, &v) in dst.iter_mut().zip(symbols) {
                        *d = table.index(v, *d);
                    }
                }
            }
            k += take;
            pos += take;
            r += 1;
        }
    }
}

/// Sites processed so far, updated once per segment.
#[derive(Debug, Default)]
pub(crate) struct Progress {
    pub(crate) forward: AtomicU64,
    pub(crate) backward: AtomicU64,
}

/// What a per-sequence pass needs besides the model and the data.
pub(crate) struct Ctx<'a> {
    pub(crate) cancel: Option<&'a AtomicBool>,
    pub(crate) progress: &'a Progress,
    pub(crate) budget: &'a AlphaCacheBudget,
}

impl Ctx<'_> {
    #[inline]
    pub(crate) fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(Cancelled.into());
        }
        Ok(())
    }
}

/// Per-thread scratch buffers, reused across the sequences of one work item.
pub(crate) struct Workspace {
    np: usize,
    obs: Vec<u16>,
    /// Forward vectors of one segment: row 0 is the vector before the segment.
    rows: Vec<f64>,
    c: Vec<f64>,
    w: Vec<f64>,
    beta: Vec<f64>,
    beta_next: Vec<f64>,
    ckpt: Vec<f64>,
}

impl Workspace {
    pub(crate) fn new(np: usize) -> Self {
        Self {
            np,
            obs: vec![0; K],
            rows: vec![0.0; (K + 1) * np],
            c: vec![0.0; K],
            w: vec![0.0; K * np],
            beta: vec![0.0; np],
            beta_next: vec![0.0; np],
            ckpt: Vec::new(),
        }
    }
}

/// Per-work-item sums: log-likelihood, expected emissions per observation
/// index, and the matrix whose product with A gives expected transitions.
#[derive(Debug, Clone)]
pub(crate) struct Partial {
    pub(crate) loglike: f64,
    /// `g[i * np + k]`: the posterior of state `k` summed over the bins of
    /// observation index `i`.
    pub(crate) g: Vec<f64>,
    pub(crate) m: Vec<f64>,
}

impl Partial {
    /// Sums for `n_obs` observation indices and `np` (padded) states.
    pub(crate) fn new(n_obs: usize, np: usize) -> Self {
        Self {
            loglike: 0.0,
            g: vec![0.0; n_obs * np],
            m: vec![0.0; np * np],
        }
    }

    pub(crate) fn merge(&mut self, other: &Partial) {
        self.loglike += other.loglike;
        for (d, s) in self.g.iter_mut().zip(&other.g) {
            *d += *s;
        }
        for (d, s) in self.m.iter_mut().zip(&other.m) {
            *d += *s;
        }
    }
}

#[inline]
fn check_norm(c: f64) -> Result<()> {
    if c.is_nan() || c <= 0.0 {
        bail!("normalization factor is zero or not finite in forward pass");
    }
    Ok(())
}

/// Full forward rows of one sequence: row `g + 1` is the forward vector at
/// site `g` (row 0 is unused), plus every normaliser.
struct FullRows {
    alpha: Vec<f64>,
    c: Vec<f64>,
}

/// Forward recursion over the whole sequence; returns the log-likelihood
/// (the sum of the log normalisers, in site order). Stores the checkpoints in
/// `ws.ckpt` when `checkpoints` is set, and every row when `full` is given.
fn forward_pass(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    checkpoints: bool,
    full: Option<&mut FullRows>,
    ctx: &Ctx<'_>,
) -> Result<f64> {
    let np = ws.np;
    let n_seg = seq.n_segments();
    let mut loglike = 0.0;
    match full {
        Some(full) => {
            for s in 0..n_seg {
                ctx.check_cancel()?;
                let (start, len) = seq.segment(s);
                seq.read(start, &mut ws.obs[..len]);
                for (t, &o) in ws.obs[..len].iter().enumerate() {
                    let g = start + t;
                    let (head, tail) = full.alpha.split_at_mut((g + 1) * np);
                    let out = &mut tail[..np];
                    let c = if g == 0 {
                        k.first(o, out)
                    } else {
                        k.forward(&head[g * np..], o, out)
                    };
                    check_norm(c)?;
                    full.c[g] = c;
                    loglike += c.ln();
                }
                ctx.progress
                    .forward
                    .fetch_add(len as u64, Ordering::Relaxed);
            }
        }
        None => {
            if checkpoints {
                ws.ckpt.clear();
                ws.ckpt.resize(n_seg * np, 0.0);
            }
            let (a, b) = ws.rows.split_at_mut(np);
            let (mut prev, mut cur) = (a, &mut b[..np]);
            for s in 0..n_seg {
                ctx.check_cancel()?;
                let (start, len) = seq.segment(s);
                seq.read(start, &mut ws.obs[..len]);
                if checkpoints && s > 0 {
                    ws.ckpt[s * np..(s + 1) * np].copy_from_slice(prev);
                }
                for (t, &o) in ws.obs[..len].iter().enumerate() {
                    let c = if start + t == 0 {
                        k.first(o, cur)
                    } else {
                        k.forward(prev, o, cur)
                    };
                    check_norm(c)?;
                    loglike += c.ln();
                    std::mem::swap(&mut prev, &mut cur);
                }
                ctx.progress
                    .forward
                    .fetch_add(len as u64, Ordering::Relaxed);
            }
        }
    }
    Ok(loglike)
}

/// Recompute the forward vectors of segment `s` from its checkpoint into
/// `rows` (row 0: the vector before the segment) and the normalisers into `c`.
/// For `s == 0` the recursion starts at the first site, needs no checkpoint,
/// and `obs` may run past the first segment.
fn recompute_segment(
    k: &Kernel,
    s: usize,
    obs: &[u16],
    ckpt: &[f64],
    rows: &mut [f64],
    c: &mut [f64],
) {
    let np = k.np();
    if s > 0 {
        rows[..np].copy_from_slice(&ckpt[s * np..(s + 1) * np]);
    }
    for (t, &o) in obs.iter().enumerate() {
        let (head, tail) = rows.split_at_mut((t + 1) * np);
        let out = &mut tail[..np];
        c[t] = if s == 0 && t == 0 {
            k.first(o, out)
        } else {
            k.forward(&head[t * np..], o, out)
        };
    }
}

/// Log-likelihood of one sequence (forward pass only, nothing stored).
pub(crate) fn log_likelihood(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    ctx: &Ctx<'_>,
) -> Result<f64> {
    forward_pass(k, seq, ws, false, None, ctx)
}

/// Forward-backward over one sequence, adding its log-likelihood, expected
/// emissions and transition sums to `acc`.
pub(crate) fn e_step(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    acc: &mut Partial,
    ctx: &Ctx<'_>,
) -> Result<()> {
    let np = ws.np;
    let len = seq.len();
    let full_bytes = ((len + 1) * np + len) * std::mem::size_of::<f64>();
    // With the O(n) kernel, recomputing a segment costs ~15% of a
    // single-threaded E-step; keeping 8 * n bytes per site for that is not
    // worth it (and its page faults and memory traffic make multi-threaded
    // runs no faster).
    let reservation = if k.cheap_forward() {
        None
    } else {
        ctx.budget.try_reserve(full_bytes)
    };
    let mut full = reservation.as_ref().map(|_| FullRows {
        alpha: vec![0.0; (len + 1) * np],
        c: vec![0.0; len],
    });

    acc.loglike += forward_pass(k, seq, ws, full.is_none(), full.as_mut(), ctx)?;

    let Workspace {
        obs,
        rows: seg_rows,
        c: seg_c,
        w,
        beta,
        beta_next,
        ckpt,
        ..
    } = ws;
    beta[..k.n_states()].fill(1.0);
    beta[k.n_states()..].fill(0.0);
    for s in (0..seq.n_segments()).rev() {
        ctx.check_cancel()?;
        let (start, len) = seq.segment(s);
        let obs = &mut obs[..len];
        seq.read(start, obs);
        let (rows, c): (&[f64], &[f64]) = match &full {
            Some(f) => (
                &f.alpha[start * np..(start + len + 1) * np],
                &f.c[start..start + len],
            ),
            None => {
                recompute_segment(k, s, obs, ckpt, seg_rows, seg_c);
                (&seg_rows[..(len + 1) * np], &seg_c[..len])
            }
        };
        // Site 0 of a sequence has no incoming transition.
        let t_first = usize::from(s == 0);
        for t in (0..len).rev() {
            let o = obs[t];
            let alpha_t = &rows[(t + 1) * np..(t + 2) * np];
            let at = usize::from(o) * np;
            let g = &mut acc.g[at..at + np];
            for ((gk, &a), &b) in g.iter_mut().zip(alpha_t).zip(beta.iter()) {
                *gk += a * b;
            }
            if t < t_first {
                break;
            }
            k.backward(beta, o, 1.0 / c[t], &mut w[t * np..(t + 1) * np], beta_next);
            std::mem::swap(beta, beta_next);
        }
        k.accumulate_transitions(
            &rows[t_first * np..],
            &w[t_first * np..],
            len - t_first,
            &mut acc.m,
        );
        ctx.progress
            .backward
            .fetch_add(len as u64, Ordering::Relaxed);
    }
    drop(full);
    drop(reservation);
    Ok(())
}

/// Checkpoints for posterior decoding of one sequence.
pub(crate) struct DecodeCheckpoints {
    /// Forward vector before each segment (row 0 unused).
    alpha: Vec<f64>,
    /// Backward vector at the last site of each segment, up to a constant
    /// factor (all ones for the last segment).
    beta: Vec<f64>,
}

/// Forward and backward checkpoint passes for posterior decoding; the two
/// passes are independent and run concurrently.
pub(crate) fn decode_checkpoints(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ctx: &Ctx<'_>,
) -> Result<DecodeCheckpoints> {
    let np = k.np();
    let (fwd, bwd) = rayon::join(
        || -> Result<Vec<f64>> {
            let mut ws = Workspace::new(np);
            forward_pass(k, seq, &mut ws, true, None, ctx)?;
            Ok(ws.ckpt)
        },
        || -> Result<Vec<f64>> {
            let n_seg = seq.n_segments();
            let mut out = vec![0.0; n_seg * np];
            let mut obs = [0u16; K];
            let mut w = vec![0.0; np];
            let mut beta = vec![0.0; np];
            let mut beta_next = vec![0.0; np];
            beta[..k.n_states()].fill(1.0);
            for s in (0..n_seg).rev() {
                ctx.check_cancel()?;
                out[s * np..(s + 1) * np].copy_from_slice(&beta);
                let (start, len) = seq.segment(s);
                seq.read(start, &mut obs[..len]);
                for t in (usize::from(s == 0)..len).rev() {
                    k.backward_normalised(&beta, obs[t], &mut w, &mut beta_next);
                    std::mem::swap(&mut beta, &mut beta_next);
                }
                ctx.progress
                    .backward
                    .fetch_add(len as u64, Ordering::Relaxed);
            }
            Ok(out)
        },
    );
    Ok(DecodeCheckpoints {
        alpha: fwd?,
        beta: bwd?,
    })
}

/// Walk backward over sites `0..obs.len()`, turning forward vectors into
/// posterior state probabilities in place: row `t + 1` of `rows` holds the
/// forward vector of site `t` on entry and its posterior on return. `beta` is
/// the normalised backward vector of the last site; `beta_next` and `w` are
/// scratch.
fn posteriors_in_place<'b>(
    k: &Kernel,
    obs: &[u16],
    rows: &mut [f64],
    mut beta: &'b mut [f64],
    mut beta_next: &'b mut [f64],
    w: &mut [f64],
) -> Result<()> {
    let np = k.np();
    for t in (0..obs.len()).rev() {
        let p = &mut rows[(t + 1) * np..(t + 2) * np];
        let mut norm = 0.0;
        for (d, &b) in p.iter_mut().zip(beta.iter()) {
            *d *= b;
            norm += *d;
        }
        if norm.is_nan() || norm <= 0.0 {
            bail!("normalization factor is zero in TMRCA posterior");
        }
        let inv = 1.0 / norm;
        for d in p.iter_mut() {
            *d *= inv;
        }
        if t > 0 {
            k.backward_normalised(beta, obs[t], w, beta_next);
            std::mem::swap(&mut beta, &mut beta_next);
        }
    }
    Ok(())
}

/// Posterior state probabilities of the sites of segment `s`, from the
/// checkpoints. Returns the segment's observation indices and its
/// posteriors (`np` values per site, padded entries zero), both held in `ws`.
pub(crate) fn segment_posteriors<'w>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ckpt: &DecodeCheckpoints,
    s: usize,
    ws: &'w mut Workspace,
) -> Result<(&'w [u16], &'w [f64])> {
    let np = ws.np;
    let (start, len) = seq.segment(s);
    let Workspace {
        obs,
        rows,
        c,
        w,
        beta,
        beta_next,
        ..
    } = ws;
    let obs = &mut obs[..len];
    seq.read(start, obs);
    let obs: &'w [u16] = obs;
    recompute_segment(k, s, obs, &ckpt.alpha, rows, c);
    beta.copy_from_slice(&ckpt.beta[s * np..(s + 1) * np]);
    posteriors_in_place(k, obs, rows, beta, beta_next, &mut w[..np])?;
    Ok((obs, &rows[np..(len + 1) * np]))
}

/// Like [`segment_posteriors`] for every site of `seq` at once, from its full
/// forward rows instead of checkpoints: one forward and one backward pass,
/// with bit-identical results (the same recursions run on the same inputs).
/// For short sequences: `ws` grows to hold `seq.len() + 1` forward vectors.
pub(crate) fn sequence_posteriors<'w>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &'w mut Workspace,
) -> Result<(&'w [u16], &'w [f64])> {
    let np = ws.np;
    let len = seq.len();
    let Workspace {
        obs,
        rows,
        c,
        w,
        beta,
        beta_next,
        ..
    } = ws;
    if obs.len() < len {
        obs.resize(len, 0);
        c.resize(len, 0.0);
        rows.resize((len + 1) * np, 0.0);
    }
    let obs = &mut obs[..len];
    seq.read(0, obs);
    let obs: &'w [u16] = obs;
    recompute_segment(k, 0, obs, &[], rows, c);
    for &ct in &c[..len] {
        check_norm(ct)?;
    }
    beta[..k.n_states()].fill(1.0);
    beta[k.n_states()..].fill(0.0);
    posteriors_in_place(k, obs, rows, beta, beta_next, &mut w[..np])?;
    Ok((obs, &rows[np..(len + 1) * np]))
}
