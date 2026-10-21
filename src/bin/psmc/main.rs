//! The `psmc-rs` program: inference (the default command), `convert`, `qc`
//! and the TUI.

mod cli;
mod console;
mod convert;
mod dataset;
mod output;
mod qc;
mod run;
mod tui;

use anyhow::Result;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use crate::cli::{Cli, Command, InferArgs, USAGE_EXIT, UsageError};
use crate::console::Console;

/// Inference on the command line. With the QC JSON on stdout, the result
/// lines go to stderr.
fn infer(args: &InferArgs) -> Result<ExitCode> {
    let json_to_stdout = args.qc_json.as_deref().is_some_and(output::is_stdout);
    let console = Console::new(args.quiet, !args.no_progress).results_on_stderr(json_to_stdout);
    let never = AtomicBool::new(false);
    let result = run::run_inference(args, &never, &|ev| console.emit(ev), !args.no_progress);
    console.finish();
    result.map(|()| ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let cli = Cli::try_parse_args(std::env::args_os()).unwrap_or_else(|e| e.exit());
    let result = match &cli.command {
        Some(Command::Convert(args)) => convert::run(args),
        Some(Command::Qc(args)) => qc::run(args),
        None if cli.infer.tui => tui::run_tui(&cli.infer).map(|()| ExitCode::SUCCESS),
        None => infer(&cli.infer),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            console::print_error(&e);
            if UsageError::is_in(&e) {
                ExitCode::from(USAGE_EXIT)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// A fresh directory of its own under the system temp dir, removed when
    /// dropped.
    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("psmc_rs_{name}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl std::ops::Deref for TempDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `dir/in.psmcfa`: two sequences of `repeat` times the same 49 bins (5
    /// heterozygous, 2 missing).
    pub fn psmcfa(dir: &Path, repeat: usize) -> PathBuf {
        let path = dir.join("in.psmcfa");
        let bins = "TTTTKTTTTTTTNNTTKTTTTTTTTKTTTTTTTTTTKTTTTTTTTTKTT".repeat(repeat);
        fs::write(&path, format!(">a\n{bins}\n>b\n{bins}\n")).unwrap();
        path
    }
}
