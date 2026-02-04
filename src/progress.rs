use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::time::Duration;

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold} {msg:.bold} [{elapsed_precise}] {bar:48.cyan/blue} {pos:>7}/{len:7} {percent:>3}% ETA {eta_precise}",
    )
    .unwrap()
    .progress_chars("█▇▆▅▄▃▂▁ ")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold} {spinner:.magenta} {msg:.bold} [{elapsed_precise}]")
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
}

pub fn bar(len: u64, prefix: &str, msg: &str) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr_with_hz(15));
    pb.set_style(bar_style());
    pb.set_prefix(prefix.to_string());
    pb.set_message(msg.to_string());
    pb
}

pub fn spinner(prefix: &str, msg: &str) -> ProgressBar {
    let pb = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(15));
    pb.set_style(spinner_style());
    pb.set_prefix(prefix.to_string());
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}
