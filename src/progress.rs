use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} {msg:.bold} [{elapsed_precise}] {bar:48.cyan/blue} {pos:>7}/{len:7} {percent:>3}% ETA {eta_precise}",
    )
    .unwrap()
    .progress_chars("█▇▆▅▄▃▂▁ ")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} {spinner:.magenta} {msg:.bold} [{elapsed_precise}]",
    )
    .unwrap()
    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
}

fn styled(pb: ProgressBar, style: ProgressStyle, prefix: &str, msg: &str) -> ProgressBar {
    pb.set_style(style);
    pb.set_prefix(prefix.to_string());
    pb.set_message(msg.to_string());
    pb
}

/// A progress bar of `len` steps drawn on stderr.
pub fn bar(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr_with_hz(15));
    styled(pb, bar_style(), prefix, msg)
}

/// A progress bar styled like [`bar`], for a caller that chooses where it
/// is drawn.
pub fn bar_raw(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    styled(ProgressBar::new(len), bar_style(), prefix, msg)
}

/// A spinner for a caller that draws it itself (for example in an
/// `indicatif::MultiProgress`) and starts its ticks.
pub fn spinner_raw(prefix: &str, msg: &str) -> ProgressBar {
    styled(ProgressBar::new_spinner(), spinner_style(), prefix, msg)
}
