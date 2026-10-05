//! The `psmc-rs` program: inference (the default command), `convert` and
//! the TUI.

mod cli;
mod console;
mod convert;
mod dataset;
mod run;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use crate::cli::{Cli, Command, InferArgs};
use crate::console::Console;

/// Inference on the command line.
fn infer(args: &InferArgs) -> Result<ExitCode> {
    let console = Console::new(args.quiet, !args.no_progress);
    let never = AtomicBool::new(false);
    let result = run::run_inference(args, &never, &|ev| console.emit(ev), !args.no_progress);
    console.finish();
    result.map(|_| ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Some(Command::Convert(args)) => convert::run(args),
        None if cli.infer.tui => tui::run_tui(&cli.infer).map(|()| ExitCode::SUCCESS),
        None => infer(&cli.infer),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
