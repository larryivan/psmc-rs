//! The progress bars the library draws on stderr when asked to.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// A progress bar of `len` steps drawn on stderr, redrawn at most 15 times a
/// second like the binary's console.
pub(crate) fn bar(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "{prefix:.bold} {msg:.bold} [{elapsed_precise}] {bar:48.cyan/blue} {pos:>7}/{len:7} {percent:>3}% ETA {eta_precise}",
    )
    .expect("a valid template")
    .progress_chars("█▇▆▅▄▃▂▁ ");
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr_with_hz(15))
        .with_style(style);
    // The setters draw, unlike `with_prefix` and `with_message`: a bar shows
    // at 0% as soon as it is made.
    pb.set_prefix(prefix.to_string());
    pb.set_message(msg.to_string());
    pb
}
