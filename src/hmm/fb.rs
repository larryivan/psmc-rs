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
use super::{CHECKPOINT_STRIDE, Cancelled};

const K: usize = CHECKPOINT_STRIDE;

/// Observation class of a raw psmcfa symbol: 0 (`T`), 1 (`K`), 2 (`N` or other).
#[inline]
pub(crate) fn obs_class(v: u8) -> u8 {
    if v <= 1 { v } else { 2 }
}

/// The observations of one sequence: one or more rows chained into one HMM chain.
pub(crate) struct SeqObs<'a> {
    rows: &'a [Vec<u8>],
    /// `offsets[r]` is the index of the first site of row `r`; the last entry is the length.
    offsets: Vec<usize>,
}

impl<'a> SeqObs<'a> {
    /// `rows` must be non-empty and contain no empty row.
    pub(crate) fn new(rows: &'a [Vec<u8>]) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut pos = 0usize;
        offsets.push(0);
        for r in rows {
            pos += r.len();
            offsets.push(pos);
        }
        Self { rows, offsets }
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

    /// Observation classes of sites `start..start + out.len()`.
    pub(crate) fn read(&self, start: usize, out: &mut [u8]) {
        let mut r = self.offsets.partition_point(|&o| o <= start) - 1;
        let mut pos = start;
        let mut k = 0;
        while k < out.len() {
            let row = &self.rows[r];
            let off = pos - self.offsets[r];
            let take = (row.len() - off).min(out.len() - k);
            for (d, &v) in out[k..k + take].iter_mut().zip(&row[off..off + take]) {
                *d = obs_class(v);
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
    obs: Vec<u8>,
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
/// class, and the matrix whose product with A gives expected transitions.
#[derive(Debug, Clone)]
pub(crate) struct Partial {
    pub(crate) loglike: f64,
    pub(crate) gobs: [Vec<f64>; 3],
    pub(crate) m: Vec<f64>,
}

impl Partial {
    pub(crate) fn new(np: usize) -> Self {
        Self {
            loglike: 0.0,
            gobs: [vec![0.0; np], vec![0.0; np], vec![0.0; np]],
            m: vec![0.0; np * np],
        }
    }

    pub(crate) fn merge(&mut self, other: &Partial) {
        self.loglike += other.loglike;
        for (dst, src) in self.gobs.iter_mut().zip(&other.gobs) {
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
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
fn recompute_segment(
    k: &Kernel,
    s: usize,
    obs: &[u8],
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
    let reservation = ctx.budget.try_reserve(full_bytes);
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
            let g = &mut acc.gobs[o as usize];
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
    /// Backward vector, normalised to sum 1, at the last site of each segment.
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
            let mut obs = [0u8; K];
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

/// Posterior state probabilities of the sites of segment `s`, written as
/// rows of `post` (`np` per site; padded entries are zero).
pub(crate) fn segment_posteriors(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ckpt: &DecodeCheckpoints,
    s: usize,
    ws: &mut Workspace,
    post: &mut [f64],
) -> Result<()> {
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
    recompute_segment(k, s, obs, &ckpt.alpha, rows, c);
    beta.copy_from_slice(&ckpt.beta[s * np..(s + 1) * np]);
    for t in (0..len).rev() {
        let alpha_t = &rows[(t + 1) * np..(t + 2) * np];
        let p = &mut post[t * np..(t + 1) * np];
        let mut norm = 0.0;
        for ((d, &a), &b) in p.iter_mut().zip(alpha_t).zip(beta.iter()) {
            *d = a * b;
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
            k.backward_normalised(beta, obs[t], &mut w[..np], beta_next);
            std::mem::swap(beta, beta_next);
        }
    }
    Ok(())
}

/// Observation classes of segment `s` (for writers that need them alongside posteriors).
pub(crate) fn segment_obs(seq: &SeqObs<'_>, s: usize, out: &mut Vec<u8>) {
    let (start, len) = seq.segment(s);
    out.resize(len, 0);
    seq.read(start, out);
}
