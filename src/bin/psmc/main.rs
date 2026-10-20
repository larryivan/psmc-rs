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
    result.map(|_| ExitCode::SUCCESS)
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
