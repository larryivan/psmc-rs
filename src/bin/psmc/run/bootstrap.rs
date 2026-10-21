//! The bootstrap of a run: its plan, checked before any work is done, the
//! replicates' fit, and the files of an earlier bootstrap in its directory.

use anyhow::{Context as _, Result};
use std::fs;
use std::path::{Path, PathBuf};

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, BootstrapSummary, bootstrap_trunk_count,
    default_bootstrap_dir, observations_to_sequences, run_bootstrap_sequences,
    summarize_bootstrap_models,
};
use psmc_rs::io::{Observations, Sequences};
use psmc_rs::opt::MStepConfig;
use psmc_rs::qc::fmt::{count, counted};

use super::{Context, Event, Stage, psmc_plot_command};
use crate::cli::{InferArgs, usage_bail};
use crate::output::{BOOTSTRAP_SUMMARY_FILES, write_bootstrap_summary, write_replicate_psmc};

/// Trunk counts below this make coarse bootstrap bands.
const FEW_TRUNKS: usize = 20;

/// The bootstrap a run asks for.
pub(super) struct BootstrapPlan {
    replicates: usize,
    pub(super) trunk_size: usize,
    seed: u64,
    pub(super) em_iters: usize,
    pub(super) dir: PathBuf,
}

impl BootstrapPlan {
    /// The bootstrap of `cli`, if any, checked before any work is done;
    /// bootstrap options that cannot apply are usage errors.
    pub(super) fn of(cli: &InferArgs, n_iter: usize, output: &Path) -> Result<Option<Self>> {
        if cli.bootstrap == 0 {
            for (set, flag) in [
                (cli.bootstrap_iters.is_some(), "--bootstrap-iters"),
                (cli.bootstrap_dir.is_some(), "--bootstrap-dir"),
            ] {
                if set {
                    usage_bail!(
                        "{flag} applies to a bootstrap; add --bootstrap N to run N replicates"
                    );
                }
            }
            return Ok(None);
        }
        let em_iters = cli.bootstrap_iters.unwrap_or(n_iter);
        if em_iters == 0 {
            usage_bail!(
                "the bootstrap replicates would run 0 EM iterations; set N_ITER above 0 or give \
                 --bootstrap-iters"
            );
        }
        Ok(Some(Self {
            replicates: cli.bootstrap,
            trunk_size: cli.bootstrap_block_size,
            seed: cli.bootstrap_seed,
            em_iters,
            dir: cli
                .bootstrap_dir
                .clone()
                .unwrap_or_else(|| default_bootstrap_dir(output)),
        }))
    }

    /// The input as sequences to resample, with the number of trunks they
    /// form, announced with what limits the bootstrap; an input too short
    /// to vary is an error, before the main fit rather than after it.
    pub(super) fn resample(
        &self,
        obs: &Observations,
        cli: &InferArgs,
        ctx: Context,
    ) -> Result<(Sequences, usize)> {
        let seqs = observations_to_sequences(obs)?;
        let trunks = bootstrap_trunk_count(&seqs.bins, self.trunk_size)
            .context("--bootstrap-block-size is too large for this input")?;
        ctx.info(format!(
            "Bootstrap: {} of {}, resampling {} of {} bins (seed {})",
            counted(self.replicates, "replicate", "replicates"),
            counted(self.em_iters, "EM iteration", "EM iterations"),
            counted(trunks, "trunk", "trunks"),
            count(self.trunk_size as u64),
            self.seed
        ));
        if let Some(path) = &cli.init_params {
            ctx.warn(format!(
                "every bootstrap replicate starts from the parameters in {}, as the main fit \
                 does (and C psmc -i -b); they were fitted to the whole input, so replicates \
                 that run few EM iterations ({} here) stay close to them and the bands \
                 understate the uncertainty further. For bands of the whole estimator, \
                 bootstrap a fit from a fresh start",
                path.display(),
                self.em_iters
            ));
        }
        if trunks < FEW_TRUNKS {
            ctx.warn(format!(
                "the input forms only {trunks} bootstrap trunks of {} bins, so replicates \
                 differ only in which of them they draw and the bands are coarse; a \
                 smaller --bootstrap-block-size gives more trunks",
                count(self.trunk_size as u64)
            ));
        }
        Ok((seqs, trunks))
    }

    /// The warning of a bootstrap whose replicates ran another number of EM
    /// iterations than the main fit's `n_iter`. psmc_plot.pl -n takes the
    /// same round of every run in its input, so the replicates cannot share
    /// a plot command with the main fit's `.psmc` output (as concatenated
    /// for C psmc's bootstrap): with fewer rounds it fails, with more it
    /// plots them at an intermediate round.
    pub(super) fn plot_warning(&self, cli: &InferArgs, n_iter: usize) -> Option<String> {
        if self.em_iters == n_iter {
            return None;
        }
        let replicates = self.dir.join("replicate_*.psmc");
        Some(format!(
            "the bootstrap replicates ran {} and the main fit {n_iter}; psmc_plot.pl -n takes the \
             same round of every run it reads, so do not plot the replicates in one file with the \
             main fit's .psmc output, but on their own: {}",
            counted(self.em_iters, "EM iteration", "EM iterations"),
            psmc_plot_command(cli, self.em_iters, &replicates.display().to_string())
        ))
    }
}

/// A bootstrap ready to run once the main start is known.
pub(super) struct Bootstrap {
    pub(super) plan: BootstrapPlan,
    /// The input as sequences, to resample.
    pub(super) seqs: Sequences,
    /// The number of trunks they form.
    pub(super) trunks: usize,
    pub(super) cfg: BootstrapConfig,
    /// The saved parameters every replicate starts from (`--init-params`,
    /// as C psmc -i -b); `None` for fresh starts.
    pub(super) start: Option<PsmcModel>,
}

impl Bootstrap {
    /// The bootstrap of `plan` on the resampled input `(seqs, trunks)`. The
    /// replicates start where the main fit starts, `model`, as C psmc's do:
    /// from a fresh start at its t_max and theta0/rho0 (-b), or from the
    /// saved parameters (-i with -b).
    pub(super) fn new(
        plan: BootstrapPlan,
        (seqs, trunks): (Sequences, usize),
        model: &PsmcModel,
        cli: &InferArgs,
    ) -> Self {
        let cfg = BootstrapConfig {
            replicates: plan.replicates,
            trunk_size: plan.trunk_size,
            seed: plan.seed,
            em_iters: plan.em_iters,
            t_max: model.t_max,
            tr_ratio: model.theta / model.rho,
            batch_size: cli.input.batch_size,
            out_dir: Some(plan.dir.clone()),
        };
        Self {
            cfg,
            start: cli.init_params.is_some().then(|| model.clone()),
            plan,
            seqs,
            trunks,
        }
    }

    /// Run the bootstrap of the fitted `model` and write its files: each
    /// replicate's JSON and `.psmc` output, and the summary of the curves,
    /// scaled by `(bin_size, gen_years)`.
    pub(super) fn run(
        &self,
        model: &PsmcModel,
        mstep_config: &MStepConfig,
        (bin_size, gen_years): (f64, f64),
        ctx: Context,
    ) -> Result<BootstrapSummary> {
        let cfg = &self.cfg;
        let dir = self.plan.dir.as_path();
        let progress = |done: usize| Event::Progress {
            stage: Stage::Bootstrap,
            done: done as u64,
            total: cfg.replicates as u64,
            iter: None,
        };
        if let Some(msg) = remove_earlier_bootstrap(dir, cfg.replicates)? {
            ctx.warn(msg);
        }
        (ctx.sink)(progress(0));
        let on_done = |n: usize| (ctx.sink)(progress(n));
        // A fresh start can be invalid for some replicates (the start's
        // checks named them); saved parameters are the main fit's start,
        // checked before it.
        let opts = BootstrapOptions {
            cancel: Some(ctx.cancel),
            thread_pool: Some(ctx.pool),
            on_done: Some(&on_done),
            start: self.start.as_ref(),
            skip_invalid_starts: self.start.is_none(),
        };
        let replicates = run_bootstrap_sequences(&self.seqs, model, cfg, mstep_config, opts)?;
        write_replicate_psmc(dir, &replicates)?;
        let mut fitted = vec![false; cfg.replicates];
        for r in &replicates {
            fitted[r.index] = true;
        }
        let skipped: Vec<usize> = (1..=cfg.replicates).filter(|&n| !fitted[n - 1]).collect();
        let models: Vec<PsmcModel> = replicates.into_iter().map(|r| r.model).collect();
        let mut summary = summarize_bootstrap_models(
            model,
            &models,
            bin_size,
            gen_years,
            cfg.trunk_size,
            cfg.seed,
        )?;
        let skipped_note = if skipped.is_empty() {
            String::new()
        } else {
            format!(
                "; {} of {}, {} skipped",
                summary.n_replicates,
                cfg.replicates,
                list_replicates(&skipped)
            )
        };
        summary.skipped_replicates = skipped;
        let (summary_tsv, summary_json) = write_bootstrap_summary(dir, &summary)?;
        ctx.output(format!(
            "Bootstrap replicates (replicate_NNN.json, replicate_NNN.psmc{skipped_note}): {}",
            dir.display()
        ));
        ctx.output(format!(
            "Bootstrap summary: {}, {}",
            summary_tsv.display(),
            summary_json.display()
        ));
        Ok(summary)
    }
}

/// Replicates by number: "replicate 3", "replicates 3 and 7", or the first
/// five and how many more ("replicates 2, 3, 4, 5, 6 and 4 more").
pub(super) fn list_replicates(numbers: &[usize]) -> String {
    let shown: Vec<String> = numbers.iter().take(5).map(usize::to_string).collect();
    match (numbers.len(), shown.as_slice()) {
        (1, [one]) => format!("replicate {one}"),
        (n, [rest @ .., last]) if n <= 5 => format!("replicates {} and {last}", rest.join(", ")),
        (n, _) => format!(
            "replicates {} and {} more",
            shown.join(", "),
            n - shown.len()
        ),
    }
}

/// The warning for a run without a bootstrap whose default bootstrap
/// directory holds the replicates of an earlier run.
pub(super) fn earlier_bootstrap_warning(output: &Path) -> Option<String> {
    let dir = default_bootstrap_dir(output);
    let mut files = replicate_files(&dir).ok()?;
    files.next().is_some().then(|| {
        format!(
            "{} holds the bootstrap of an earlier run, not of this fit; this run made no \
             bootstrap",
            dir.display()
        )
    })
}

/// The replicate files in `dir` (`replicate_NNN.json` and
/// `replicate_NNN.psmc`, as a bootstrap writes them) with their replicate
/// numbers; none when `dir` does not exist.
fn replicate_files(dir: &Path) -> Result<impl Iterator<Item = (usize, PathBuf)>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => Some(entries),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("failed to list {}", dir.display())),
    };
    Ok(entries.into_iter().flatten().filter_map(|entry| {
        let entry = entry.ok()?;
        if !entry.file_type().ok()?.is_file() {
            return None;
        }
        let name = entry.file_name();
        let name = name.to_str()?;
        let stem = name
            .strip_suffix(".json")
            .or_else(|| name.strip_suffix(".psmc"))?;
        let number = stem.strip_prefix("replicate_")?;
        if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some((number.parse().ok()?, entry.path()))
    }))
}

/// Remove the files an earlier bootstrap left in `dir` (its replicate
/// files and its summary) before this run writes any, and say so for the
/// replicates beyond this run's `replicates`, which this run would not
/// have replaced.
///
/// This run writes each replicate's JSON as the replicate finishes, the
/// `.psmc` files and the summary only at the end. Replacing the earlier
/// files one by one would leave the two runs mixed when this one stops
/// part way (an error, a kill, running out of memory), and would leave
/// replicates beyond this run's count in place for good, which
/// `cat replicate_*.psmc` (the README's recipe for psmc_plot.pl) and globs
/// of `replicate_*.json` would then mix with this run's while the summary
/// and the report use only the new ones.
fn remove_earlier_bootstrap(dir: &Path, replicates: usize) -> Result<Option<String>> {
    let earlier: Vec<(usize, PathBuf)> = replicate_files(dir)?.collect();
    let summary = BOOTSTRAP_SUMMARY_FILES
        .map(|name| dir.join(name))
        .into_iter()
        .filter(|path| path.is_file());
    for path in earlier.iter().map(|(_, path)| path.clone()).chain(summary) {
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    // One number per file: a replicate's JSON and .psmc count twice.
    let stale: Vec<usize> = earlier
        .iter()
        .map(|&(n, _)| n)
        .filter(|&n| n > replicates)
        .collect();
    let (Some(&first), Some(&last)) = (stale.iter().min(), stale.iter().max()) else {
        return Ok(None);
    };
    let which = if first == last {
        format!("replicate {first}")
    } else {
        format!("replicates {first} to {last}")
    };
    Ok(Some(format!(
        "{} held {which} of an earlier bootstrap, beyond this run's {replicates}; their {} \
         removed so that the directory holds only this run's replicates",
        dir.display(),
        counted(stale.len(), "file was", "files were")
    )))
}
