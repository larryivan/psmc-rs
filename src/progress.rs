//! The progress bars the library draws on stderr when asked to.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// A progress bar of `len` steps drawn on stderr.
pub(crate) fn bar(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr_with_hz(15));
    styled(pb, prefix, msg)
}

/// [`bar`] redrawn at indicatif's default rate.
pub(crate) fn bar_raw(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    styled(ProgressBar::new(len), prefix, msg)
}

fn styled(pb: ProgressBar, prefix: &str, msg: &str) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "{prefix:.bold} {msg:.bold} [{elapsed_precise}] {bar:48.cyan/blue} {pos:>7}/{len:7} {percent:>3}% ETA {eta_precise}",
    )
    .expect("a valid template")
    .progress_chars("█▇▆▅▄▃▂▁ ");
    pb.with_style(style)
        .with_prefix(prefix.to_string())
        .with_message(msg.to_string())
}
