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

use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::budget::AlphaCacheBudget;
use super::kernel::Kernel;
use super::obs::{ObsTable, obs_class};
use super::simd::dispatch;
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
#[inline(always)]
fn forward_pass<const FUSED: bool>(
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
    forward_pass::<FUSED>(k, seq, ws, false, None, ctx)
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

    acc.loglike += forward_pass::<FUSED>(k, seq, ws, full.is_none(), full.as_mut(), ctx)?;

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
                recompute_segment::<FUSED>(k, s, obs, ckpt, seg_rows, seg_c);
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
            k.backward::<FUSED>(beta, o, 1.0 / c[t], &mut w[t * np..(t + 1) * np], beta_next);
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
    forward_pass::<FUSED>(k, seq, &mut ws, true, None, ctx)?;
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
        let mut rng = SmallRng::seed_from_u64(11);
        [5 * K + 17, 2 * K, 3]
            .iter()
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

    /// Calls `$imp::<true>` or `$imp::<false>` as `$fused` says.
    macro_rules! instance {
        ($fused:expr, $imp:ident($($arg:expr),* $(,)?)) => {
            if $fused { $imp::<true>($($arg),*) } else { $imp::<false>($($arg),*) }
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
        let ll = match run {
            Some(f) => instance!(f, log_likelihood_impl(k, &seq, &mut ws, &ctx)),
            None => log_likelihood(k, &seq, &mut ws, &ctx),
        };
        let mut out = vec![vec![ll.unwrap()]];
        match run {
            Some(f) => instance!(f, e_step_impl(k, &seq, &mut ws, &mut acc, &ctx)),
            None => e_step(k, &seq, &mut ws, &mut acc, &ctx),
        }
        .unwrap();
        out.extend([vec![acc.loglike], acc.g, acc.m]);
        let ckpt = DecodeCheckpoints {
            alpha: match run {
                Some(f) => instance!(f, forward_checkpoints_impl(k, &seq, &ctx)),
                None => forward_checkpoints(k, &seq, &ctx),
            }
            .unwrap(),
            beta: match run {
                Some(f) => instance!(f, backward_checkpoints_impl(k, &seq, &ctx)),
                None => backward_checkpoints(k, &seq, &ctx),
            }
            .unwrap(),
        };
        for s in 0..seq.n_segments() {
            let (_, post) = match run {
                Some(f) => instance!(f, segment_posteriors_impl(k, &seq, &ckpt, s, &mut ws)),
                None => segment_posteriors(k, &seq, &ckpt, s, &mut ws),
            }
            .unwrap();
            out.push(post.to_vec());
        }
        let (_, post) = match run {
            Some(f) => instance!(f, sequence_posteriors_impl(k, &seq, &mut ws)),
            None => sequence_posteriors(k, &seq, &mut ws),
        }
        .unwrap();
        out.push(post.to_vec());
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
}
