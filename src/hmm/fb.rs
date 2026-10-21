//! The per-sequence forward-backward engine shared by the E-step, the
//! log-likelihood and posterior decoding.
//!
//! Memory is bounded by checkpointing: the forward pass keeps the forward
//! vector entering every segment of [`CHECKPOINT_STRIDE`] sites, and the
//! backward pass recomputes one segment's forward vectors at a time from its
//! checkpoint. Recomputation runs the same kernel on the same inputs, so the
//! recomputed vectors are bit-identical to the ones the forward pass produced,
//! and results do not depend on whether full rows were cached instead.
//!
//! The passes are compiled twice on x86-64 ([`super::simd`]), so everything
//! they run per site is `#[inline(always)]`, and generic over `FUSED`, the
//! choice of fused or unfused multiply-adds that each build makes at
//! compile time.
//!
//! While the pool has threads with no sequence to process, the E-step's
//! backward pass over a long sequence takes some of them
//! ([`backward_blocks`]), with the same results to the bit.

use anyhow::{Result, bail};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::budget::AlphaCacheBudget;
use super::kernel::Kernel;
use super::obs::{ObsTable, obs_class};
use super::simd::dispatch;
use super::{CHECKPOINT_STRIDE, Cancelled, sched};
use crate::io::BinCounts;

const K: usize = CHECKPOINT_STRIDE;

/// Segments per block of [`backward_blocks`].
const BLOCK: usize = 16;

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
    /// The forward vector entering each segment of a sequence (row 0
    /// unused), when its forward pass keeps them.
    ckpt: Vec<f64>,
    /// The blocks of [`backward_blocks`], and the transition sums of each
    /// segment of one; empty until it first runs.
    blocks: [Block; 3],
    sums: Vec<f64>,
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
            blocks: Default::default(),
            sums: Vec::new(),
        }
    }
}

/// What the backward pass needs of [`BLOCK`] segments: per segment, the
/// buffers of [`Workspace`] of the same names, one after another.
#[derive(Default)]
struct Block {
    obs: Vec<u16>,
    rows: Vec<f64>,
    c: Vec<f64>,
    w: Vec<f64>,
}

impl Block {
    fn new(np: usize) -> Self {
        Self {
            obs: vec![0; BLOCK * K],
            rows: vec![0.0; BLOCK * (K + 1) * np],
            c: vec![0.0; BLOCK * K],
            w: vec![0.0; BLOCK * K * np],
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

/// What a forward pass keeps besides the log-likelihood.
enum Keep<'a> {
    Nothing,
    /// The forward vector entering each segment, in [`Workspace::ckpt`].
    Checkpoints,
    /// Every forward vector and normaliser.
    Rows(&'a mut FullRows),
}

/// Forward recursion over the whole sequence; returns the log-likelihood
/// (the sum of the log normalisers, in site order).
#[inline(always)]
fn forward_pass<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    keep: Keep<'_>,
    ctx: &Ctx<'_>,
) -> Result<f64> {
    let np = ws.np;
    let n_seg = seq.n_segments();
    let mut loglike = 0.0;
    match keep {
        Keep::Rows(full) => {
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
                        k.forward::<FUSED>(&head[g * np..], o, out)
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
        Keep::Nothing | Keep::Checkpoints => {
            let checkpoints = matches!(keep, Keep::Checkpoints);
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
                        k.forward::<FUSED>(prev, o, cur)
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
#[inline(always)]
fn recompute_segment<const FUSED: bool>(
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
            k.forward::<FUSED>(&head[t * np..], o, out)
        };
    }
}

dispatch! {
    /// Log-likelihood of one sequence (forward pass only, nothing stored).
    pub(crate) fn log_likelihood(
        k: &Kernel,
        seq: &SeqObs<'_>,
        ws: &mut Workspace,
        ctx: &Ctx<'_>,
    ) -> Result<f64> = log_likelihood_impl;
}

#[inline(always)]
fn log_likelihood_impl<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    ctx: &Ctx<'_>,
) -> Result<f64> {
    forward_pass::<FUSED>(k, seq, ws, Keep::Nothing, ctx)
}

dispatch! {
    /// Forward-backward over one sequence, adding its log-likelihood,
    /// expected emissions and transition sums to `acc`.
    pub(crate) fn e_step(
        k: &Kernel,
        seq: &SeqObs<'_>,
        ws: &mut Workspace,
        acc: &mut Partial,
        ctx: &Ctx<'_>,
    ) -> Result<()> = e_step_impl;
}

#[inline(always)]
fn e_step_impl<const FUSED: bool>(
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

    let keep = match full.as_mut() {
        Some(rows) => Keep::Rows(rows),
        None => Keep::Checkpoints,
    };
    acc.loglike += forward_pass::<FUSED>(k, seq, ws, keep, ctx)?;

    ws.beta[..k.n_states()].fill(1.0);
    ws.beta[k.n_states()..].fill(0.0);
    match full {
        // Other threads take part while some are free, for sequences of
        // more than one block.
        None if seq.n_segments() > BLOCK => {
            backward_blocks::<FUSED>(k, seq, ws, acc, ctx, sched::spare_threads)
        }
        _ => backward_segments::<FUSED>(k, seq, 0..seq.n_segments(), full.as_ref(), ws, acc, ctx),
    }
}

/// The backward pass over segments `segs` of a sequence, the last first, on
/// this thread: each segment's forward vectors come from `full` or are
/// recomputed from the checkpoints in `ws`; `ws.beta` holds the scaled
/// backward vector of the last site of the last segment, and is left at the
/// first site of the first.
#[inline(always)]
fn backward_segments<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    segs: Range<usize>,
    full: Option<&FullRows>,
    ws: &mut Workspace,
    acc: &mut Partial,
    ctx: &Ctx<'_>,
) -> Result<()> {
    let np = ws.np;
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
    for s in segs.rev() {
        ctx.check_cancel()?;
        let (start, len) = seq.segment(s);
        let obs = &mut obs[..len];
        seq.read(start, obs);
        let (rows, c): (&[f64], &[f64]) = match full {
            Some(f) => (
                &f.alpha[start * np..(start + len + 1) * np],
                &f.c[start..start + len],
            ),
            None => {
                recompute_segment::<FUSED>(k, s, obs, ckpt, seg_rows, seg_c);
                (&seg_rows[..(len + 1) * np], &seg_c[..len])
            }
        };
        backward_segment::<FUSED>(k, s, obs, rows, c, w, beta, beta_next, &mut acc.g);
        // Site 0 of a sequence has no incoming transition.
        let t_first = usize::from(s == 0);
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
    Ok(())
}

/// The backward recursion over segment `s` (observations `obs`, forward
/// vectors `rows` as [`recompute_segment`] leaves them, normalisers `c`),
/// from the scaled backward vector of its last site in `beta` to that of its
/// first: adds the posterior of each site to `g`, and leaves in `w` the
/// weights of [`Kernel::backward`] of its sites but site 0 of the sequence,
/// which has no incoming transition.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn backward_segment<const FUSED: bool>(
    k: &Kernel,
    s: usize,
    obs: &[u16],
    rows: &[f64],
    c: &[f64],
    w: &mut [f64],
    beta: &mut Vec<f64>,
    beta_next: &mut Vec<f64>,
    g: &mut [f64],
) {
    let np = k.np();
    let t_first = usize::from(s == 0);
    for t in (0..obs.len()).rev() {
        let o = obs[t];
        let alpha_t = &rows[(t + 1) * np..(t + 2) * np];
        let at = usize::from(o) * np;
        let g = &mut g[at..at + np];
        for ((gk, &a), &b) in g.iter_mut().zip(alpha_t).zip(beta.iter()) {
            *gk += a * b;
        }
        if t < t_first {
            break;
        }
        k.backward::<FUSED>(beta, o, 1.0 / c[t], &mut w[t * np..(t + 1) * np], beta_next);
        std::mem::swap(beta, beta_next);
    }
}

/// The backward pass from the checkpoints, block by block of [`BLOCK`]
/// segments, the last first. While `spare()` says that the pool has threads
/// to spare, it is pipelined ([`pipeline_step`]): this thread runs the
/// backward recursion over block `b` while the pool recomputes the forward
/// vectors of block `b - 1` and sums the transitions of block `b + 1`.
/// Otherwise a block runs as [`backward_segments`] runs it.
///
/// The recursion and its additions to `acc.g` stay on this thread, in
/// order; every kernel runs on the inputs it has in [`backward_segments`];
/// and [`Kernel::accumulate_transitions`] sums a segment on its own and adds
/// the sums to `m` at its end, so `acc.m` gets the same additions in the
/// same order. The results are the same to the bit whichever way each block
/// runs.
#[inline(always)]
fn backward_blocks<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ws: &mut Workspace,
    acc: &mut Partial,
    ctx: &Ctx<'_>,
    spare: impl Fn() -> bool,
) -> Result<()> {
    let np = ws.np;
    let n_seg = seq.n_segments();
    let segs = |b: usize| b * BLOCK..n_seg.min((b + 1) * BLOCK);
    // Whether the block after the current one was pipelined: then the
    // current one is recomputed, and the transition sums of the one after
    // are not yet in acc.m.
    let mut piped = false;
    for b in (0..n_seg.div_ceil(BLOCK)).rev() {
        if spare() {
            let Workspace {
                blocks,
                sums,
                beta,
                beta_next,
                ckpt,
                ..
            } = &mut *ws;
            if sums.is_empty() {
                *blocks = [(); 3].map(|()| Block::new(np));
                *sums = vec![0.0; BLOCK * np * np];
            }
            let [prev, cur, next] = blocks
                .get_disjoint_mut([(b + 2) % 3, b % 3, (b + 1) % 3])
                .expect("three blocks");
            let (ckpt, m): (&[f64], _) = (ckpt, &mut acc.m);
            if !piped {
                let this = Some((segs(b), &mut *cur));
                pipeline_step::<FUSED>(k, seq, ckpt, this, None, sums, m, || Ok(()))?;
            }
            let before = (b > 0).then(|| (segs(b - 1), prev));
            let after = piped.then(|| (segs(b + 1), &*next));
            let g = &mut acc.g;
            pipeline_step::<FUSED>(k, seq, ckpt, before, after, sums, m, || {
                backward_block::<FUSED>(k, seq, segs(b), cur, beta, beta_next, g, ctx)
            })?;
            piped = true;
        } else {
            if piped {
                let after = Some((segs(b + 1), &ws.blocks[(b + 1) % 3]));
                let m = &mut acc.m;
                pipeline_step::<FUSED>(k, seq, &ws.ckpt, None, after, &mut ws.sums, m, || Ok(()))?;
                piped = false;
            }
            backward_segments::<FUSED>(k, seq, segs(b), None, ws, acc, ctx)?;
        }
    }
    if piped {
        let after = Some((segs(0), &ws.blocks[0]));
        let m = &mut acc.m;
        pipeline_step::<FUSED>(k, seq, &ws.ckpt, None, after, &mut ws.sums, m, || Ok(()))?;
    }
    Ok(())
}

/// Runs `here` on this thread while the pool takes, as tasks of one segment
/// each, the recomputation of the segments of `before` into its block and
/// the transition sums of the segments of `after`, each into a matrix of
/// `sums`; then adds those to `m` from the last segment to the first.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pipeline_step<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ckpt: &[f64],
    before: Option<(Range<usize>, &mut Block)>,
    after: Option<(Range<usize>, &Block)>,
    sums: &mut [f64],
    m: &mut [f64],
    here: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let np = k.np();
    let (rl, wl, ml) = ((K + 1) * np, K * np, np * np);
    let n_after = after.as_ref().map_or(0, |(segs, _)| segs.len());
    let done = rayon::in_place_scope(|scope| {
        if let Some((segs, blk)) = before {
            let slots = blk.obs.chunks_mut(K).zip(blk.rows.chunks_mut(rl));
            for (((obs, rows), c), s) in slots.zip(blk.c.chunks_mut(K)).zip(segs) {
                scope.spawn(move |_| recompute::<FUSED>(k, seq, s, ckpt, obs, rows, c));
            }
        }
        if let Some((segs, blk)) = after {
            let slots = sums.chunks_mut(ml).zip(blk.rows.chunks(rl));
            for (((sum, rows), w), s) in slots.zip(blk.w.chunks(wl)).zip(segs) {
                let len = seq.segment(s).1;
                scope.spawn(move |_| transitions::<FUSED>(k, s, len, rows, w, sum));
            }
        }
        here()
    });
    for sum in sums[..n_after * ml].chunks(ml).rev() {
        for (d, &x) in m.iter_mut().zip(sum) {
            *d += x;
        }
    }
    done
}

dispatch! {
    /// The backward recursion over segments `segs` in `blk`, the last first,
    /// as [`backward_segments`] runs it.
    #[allow(clippy::too_many_arguments)]
    fn backward_block<const FUSED: bool>(
        k: &Kernel,
        seq: &SeqObs<'_>,
        segs: Range<usize>,
        blk: &mut Block,
        beta: &mut Vec<f64>,
        beta_next: &mut Vec<f64>,
        g: &mut [f64],
        ctx: &Ctx<'_>,
    ) -> Result<()> = backward_block_impl;
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn backward_block_impl<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    segs: Range<usize>,
    blk: &mut Block,
    beta: &mut Vec<f64>,
    beta_next: &mut Vec<f64>,
    g: &mut [f64],
    ctx: &Ctx<'_>,
) -> Result<()> {
    let np = k.np();
    let (rl, wl) = ((K + 1) * np, K * np);
    for (l, s) in segs.enumerate().rev() {
        ctx.check_cancel()?;
        let len = seq.segment(s).1;
        backward_segment::<FUSED>(
            k,
            s,
            &blk.obs[l * K..l * K + len],
            &blk.rows[l * rl..(l + 1) * rl],
            &blk.c[l * K..(l + 1) * K],
            &mut blk.w[l * wl..(l + 1) * wl],
            beta,
            beta_next,
            g,
        );
        ctx.progress
            .backward
            .fetch_add(len as u64, Ordering::Relaxed);
    }
    Ok(())
}

dispatch! {
    /// The observations, forward vectors and normalisers of segment `s`
    /// ([`recompute_segment`]), on a thread of [`pipeline_step`].
    fn recompute<const FUSED: bool>(
        k: &Kernel,
        seq: &SeqObs<'_>,
        s: usize,
        ckpt: &[f64],
        obs: &mut [u16],
        rows: &mut [f64],
        c: &mut [f64],
    ) = recompute_impl;
}

#[inline(always)]
fn recompute_impl<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    s: usize,
    ckpt: &[f64],
    obs: &mut [u16],
    rows: &mut [f64],
    c: &mut [f64],
) {
    let (start, len) = seq.segment(s);
    let obs = &mut obs[..len];
    seq.read(start, obs);
    recompute_segment::<FUSED>(k, s, obs, ckpt, rows, c);
}

dispatch! {
    /// The transition sums of segment `s` of `len` sites alone, into `sum`
    /// (as [`backward_segments`] adds them to `acc.m`), on a thread of
    /// [`pipeline_step`].
    fn transitions<const FUSED: bool>(
        k: &Kernel,
        s: usize,
        len: usize,
        rows: &[f64],
        w: &[f64],
        sum: &mut [f64],
    ) = transitions_impl;
}

#[inline(always)]
fn transitions_impl<const FUSED: bool>(
    k: &Kernel,
    s: usize,
    len: usize,
    rows: &[f64],
    w: &[f64],
    sum: &mut [f64],
) {
    let np = k.np();
    let t_first = usize::from(s == 0);
    sum.fill(0.0);
    k.accumulate_transitions(
        &rows[t_first * np..],
        &w[t_first * np..],
        len - t_first,
        sum,
    );
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
    let (fwd, bwd) = rayon::join(
        || forward_checkpoints(k, seq, ctx),
        || backward_checkpoints(k, seq, ctx),
    );
    Ok(DecodeCheckpoints {
        alpha: fwd?,
        beta: bwd?,
    })
}

dispatch! {
    /// [`DecodeCheckpoints::alpha`].
    fn forward_checkpoints(k: &Kernel, seq: &SeqObs<'_>, ctx: &Ctx<'_>) -> Result<Vec<f64>> =
        forward_checkpoints_impl;
}

#[inline(always)]
fn forward_checkpoints_impl<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ctx: &Ctx<'_>,
) -> Result<Vec<f64>> {
    let mut ws = Workspace::new(k.np());
    forward_pass::<FUSED>(k, seq, &mut ws, Keep::Checkpoints, ctx)?;
    Ok(ws.ckpt)
}

dispatch! {
    /// [`DecodeCheckpoints::beta`].
    fn backward_checkpoints(k: &Kernel, seq: &SeqObs<'_>, ctx: &Ctx<'_>) -> Result<Vec<f64>> =
        backward_checkpoints_impl;
}

#[inline(always)]
fn backward_checkpoints_impl<const FUSED: bool>(
    k: &Kernel,
    seq: &SeqObs<'_>,
    ctx: &Ctx<'_>,
) -> Result<Vec<f64>> {
    let np = k.np();
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
            k.backward_normalised::<FUSED>(&beta, obs[t], &mut w, &mut beta_next);
            std::mem::swap(&mut beta, &mut beta_next);
        }
        ctx.progress
            .backward
            .fetch_add(len as u64, Ordering::Relaxed);
    }
    Ok(out)
}

/// Walk backward over sites `0..obs.len()`, turning forward vectors into
/// posterior state probabilities in place: row `t + 1` of `rows` holds the
/// forward vector of site `t` on entry and its posterior on return. `beta` is
/// the normalised backward vector of the last site; `beta_next` and `w` are
/// scratch.
#[inline(always)]
fn posteriors_in_place<'b, const FUSED: bool>(
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
            k.backward_normalised::<FUSED>(beta, obs[t], w, beta_next);
            std::mem::swap(&mut beta, &mut beta_next);
        }
    }
    Ok(())
}

dispatch! {
    /// Posterior state probabilities of the sites of segment `s`, from the
    /// checkpoints. Returns the segment's observation indices and its
    /// posteriors (`np` values per site, padded entries zero), both held in `ws`.
    pub(crate) fn segment_posteriors<'w>(
        k: &Kernel,
        seq: &SeqObs<'_>,
        ckpt: &DecodeCheckpoints,
        s: usize,
        ws: &'w mut Workspace,
    ) -> Result<(&'w [u16], &'w [f64])> = segment_posteriors_impl;
}

#[inline(always)]
fn segment_posteriors_impl<'w, const FUSED: bool>(
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
    recompute_segment::<FUSED>(k, s, obs, &ckpt.alpha, rows, c);
    beta.copy_from_slice(&ckpt.beta[s * np..(s + 1) * np]);
    posteriors_in_place::<FUSED>(k, obs, rows, beta, beta_next, &mut w[..np])?;
    Ok((obs, &rows[np..(len + 1) * np]))
}

dispatch! {
    /// Like [`segment_posteriors`] for every site of `seq` at once, from its
    /// full forward rows instead of checkpoints: one forward and one backward
    /// pass, with bit-identical results (the same recursions run on the same
    /// inputs). For short sequences: `ws` grows to hold `seq.len() + 1`
    /// forward vectors.
    pub(crate) fn sequence_posteriors<'w>(
        k: &Kernel,
        seq: &SeqObs<'_>,
        ws: &'w mut Workspace,
    ) -> Result<(&'w [u16], &'w [f64])> = sequence_posteriors_impl;
}

#[inline(always)]
fn sequence_posteriors_impl<'w, const FUSED: bool>(
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
    recompute_segment::<FUSED>(k, 0, obs, &[], rows, c);
    for &ct in &c[..len] {
        check_norm(ct)?;
    }
    beta[..k.n_states()].fill(1.0);
    beta[k.n_states()..].fill(0.0);
    posteriors_in_place::<FUSED>(k, obs, rows, beta, beta_next, &mut w[..np])?;
    Ok((obs, &rows[np..(len + 1) * np]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PsmcModel;
    use crate::hmm::simd::{self, DEFAULT_FUSED, SimdPath};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// The PSMC model of `tests/estep_determinism.rs`: semiseparable, or
    /// with its rows perturbed and renormalised so that the dense kernels run.
    fn kernel(dense: bool) -> Kernel {
        let model =
            PsmcModel::new(15.0, 63, 2e-3, 4e-4, 2.5e-8, Some("4+25*2+4+6".to_string())).unwrap();
        let mut a = model.transition_matrix().clone();
        if dense {
            let mut rng = SmallRng::seed_from_u64(3);
            for mut row in a.rows_mut() {
                row.mapv_inplace(|v| v * (1.0 + 1e-3 * rng.gen_range(0.0..1.0)));
                let s: f64 = row.sum();
                row.mapv_inplace(|v| v / s);
            }
        }
        let k = Kernel::new(
            model.prior_matrix(),
            &a,
            model.emission_matrix(),
            &ObsTable::plain(),
        );
        assert_eq!(k.cheap_forward(), !dense);
        k
    }

    /// Rows of one sequence: mostly `T`, some `K` and `N`, over several segments.
    fn rows() -> Vec<Vec<u8>> {
        rows_of(&[5 * K + 17, 2 * K, 3])
    }

    /// Rows of `lens` sites of one sequence, as [`rows`].
    fn rows_of(lens: &[usize]) -> Vec<Vec<u8>> {
        let mut rng = SmallRng::seed_from_u64(11);
        lens.iter()
            .map(|&len| {
                (0..len)
                    .map(|_| match rng.gen_range(0..100) {
                        0 => 1,
                        1..=4 => 2,
                        _ => 0,
                    })
                    .collect()
            })
            .collect()
    }

    /// Calls `$imp::<true>` or `$imp::<false>` for `$run` `Some(fused)`, and
    /// the dispatched `$disp` for `None`.
    macro_rules! run {
        ($run:expr, $imp:ident | $disp:ident($($arg:expr),* $(,)?)) => {
            match $run {
                Some(true) => $imp::<true>($($arg),*),
                Some(false) => $imp::<false>($($arg),*),
                None => $disp($($arg),*),
            }
        };
    }

    /// Every output of every pass of `k` over `rows` (one sequence): run
    /// with the instance `Some(fused)` of the code compiled without target
    /// features, or with `None`, dispatched as the crate runs the passes.
    fn all_passes(k: &Kernel, rows: &[Vec<u8>], budget: usize, run: Option<bool>) -> Vec<Vec<f64>> {
        let table = ObsTable::plain();
        let seq = SeqObs::new(rows, None, &table);
        let progress = Progress::default();
        let budget = AlphaCacheBudget::new(budget);
        let ctx = Ctx {
            cancel: None,
            progress: &progress,
            budget: &budget,
        };
        let np = k.np();
        let mut ws = Workspace::new(np);
        let mut acc = Partial::new(table.len(), np);
        let ll = run!(
            run,
            log_likelihood_impl | log_likelihood(k, &seq, &mut ws, &ctx)
        );
        let mut out = vec![vec![ll.unwrap()]];
        run!(run, e_step_impl | e_step(k, &seq, &mut ws, &mut acc, &ctx)).unwrap();
        out.extend([vec![acc.loglike], acc.g, acc.m]);
        let ckpt = DecodeCheckpoints {
            alpha: run!(
                run,
                forward_checkpoints_impl | forward_checkpoints(k, &seq, &ctx)
            )
            .unwrap(),
            beta: run!(
                run,
                backward_checkpoints_impl | backward_checkpoints(k, &seq, &ctx)
            )
            .unwrap(),
        };
        for s in 0..seq.n_segments() {
            let post = run!(
                run,
                segment_posteriors_impl | segment_posteriors(k, &seq, &ckpt, s, &mut ws)
            );
            out.push(post.unwrap().1.to_vec());
        }
        let post = run!(
            run,
            sequence_posteriors_impl | sequence_posteriors(k, &seq, &mut ws)
        );
        out.push(post.unwrap().1.to_vec());
        out.extend([ckpt.alpha, ckpt.beta]);
        out
    }

    /// Every dispatched pass gives the bits of its own code compiled without
    /// target features: on x86-64 CPUs with AVX2 and FMA, the AVX2 and FMA
    /// build against the portable code instantiated fused (whose
    /// multiply-adds call libm's `fma`), so that vectorising changes nothing
    /// but speed; elsewhere the dispatched pass is that code itself. With
    /// the semiseparable and the dense kernels, with and without full
    /// forward rows.
    #[test]
    fn dispatched_passes_give_the_bits_of_their_instance() {
        let path = simd::simd_path();
        if path != SimdPath::Avx2Fma {
            eprintln!("this CPU runs the {path} kernels: no AVX2 and FMA build to check");
        }
        let fused = path == SimdPath::Avx2Fma || DEFAULT_FUSED;
        let rows = rows();
        for dense in [false, true] {
            let k = kernel(dense);
            for budget in [0, usize::MAX] {
                let want = all_passes(&k, &rows, budget, Some(fused));
                let got = all_passes(&k, &rows, budget, None);
                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let same = g.len() == w.len()
                        && g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        same,
                        "{path}, dense {dense}, budget {budget}: output {i} differs"
                    );
                }
            }
        }
    }

    /// The fused and unfused instances of every pass agree to a relative
    /// 1e-12: the AVX2 and FMA and the portable x86-64 kernels differ by
    /// just that. Measured: 7.8e-15 here, 1.4e-14 over 15,000 sites (the
    /// differences do not grow along the chain); the dense kernels do not
    /// use `fmadd` and agree to the bit.
    #[test]
    fn fused_and_unfused_instances_agree() {
        let rows = rows();
        for dense in [false, true] {
            let k = kernel(dense);
            let fused = all_passes(&k, &rows, 0, Some(true));
            let unfused = all_passes(&k, &rows, 0, Some(false));
            for (i, (a, b)) in fused.iter().zip(&unfused).enumerate() {
                for (x, y) in a.iter().zip(b) {
                    let close = if dense {
                        x == y
                    } else {
                        (x - y).abs() <= 1e-12 * y.abs()
                    };
                    assert!(close, "dense {dense}: output {i}: {x} fused, {y} unfused");
                }
            }
        }
    }

    /// The E-step's sums over `rows` (one sequence), as bits, with the
    /// backward pass of [`backward_blocks`] in a pool of `threads`, which
    /// pipelines the blocks for which `piped` (of their count from the last
    /// block) says so.
    fn blocks_bits<const FUSED: bool>(
        k: &Kernel,
        rows: &[Vec<u8>],
        threads: usize,
        piped: &(dyn Fn(usize) -> bool + Sync),
    ) -> Vec<u64> {
        let table = ObsTable::plain();
        let seq = SeqObs::new(rows, None, &table);
        let progress = Progress::default();
        let budget = AlphaCacheBudget::new(0);
        let ctx = Ctx {
            cancel: None,
            progress: &progress,
            budget: &budget,
        };
        let mut ws = Workspace::new(k.np());
        let mut acc = Partial::new(table.len(), k.np());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let keep = Keep::Checkpoints;
            acc.loglike = forward_pass::<FUSED>(k, &seq, &mut ws, keep, &ctx).unwrap();
            ws.beta[..k.n_states()].fill(1.0);
            let block = std::cell::Cell::new(0);
            let spare = || {
                block.set(block.get() + 1);
                piped(block.get() - 1)
            };
            backward_blocks::<FUSED>(k, &seq, &mut ws, &mut acc, &ctx, spare).unwrap();
        });
        assert_eq!(progress.backward.into_inner(), seq.len() as u64);
        let values = [&[acc.loglike][..], &acc.g, &acc.m, &ws.beta];
        values
            .iter()
            .flat_map(|v| v.iter().map(|x| x.to_bits()))
            .collect()
    }

    /// The pipelined backward pass gives the bits of the pass on one
    /// thread, whichever blocks it pipelines, on one thread or several: on
    /// sequences of a block and one site, of two whole blocks, and of
    /// chained rows ending within a segment, with the semiseparable and the
    /// dense kernels, fused and not.
    #[test]
    fn pipelined_blocks_give_the_bits_of_one_thread() {
        fn check<const FUSED: bool>(k: &Kernel, rows: &[Vec<u8>]) {
            let patterns: [(&str, &(dyn Fn(usize) -> bool + Sync)); 4] = [
                ("every block", &|_| true),
                ("every other block", &|b| b % 2 == 0),
                ("all but the first", &|b| b > 0),
                ("the first", &|b| b == 0),
            ];
            let want = blocks_bits::<FUSED>(k, rows, 1, &|_| false);
            for threads in [1, 4] {
                for (name, piped) in patterns {
                    let got = blocks_bits::<FUSED>(k, rows, threads, piped);
                    assert!(
                        got == want,
                        "fused {FUSED}, {threads} threads, {name} piped"
                    );
                }
            }
        }
        for lens in [
            &[BLOCK * K + 1][..],
            &[2 * BLOCK * K],
            &[3 * BLOCK * K + 5 * K + 17, 2 * K, 3],
        ] {
            let rows = rows_of(lens);
            for dense in [false, true] {
                let k = kernel(dense);
                check::<true>(&k, &rows);
                check::<false>(&k, &rows);
            }
        }
    }
}
