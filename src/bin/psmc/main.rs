//! The `psmc-rs` program.

mod cli;
mod console;
mod dataset;
mod run;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::sync::atomic::AtomicBool;

use crate::cli::Cli;
use crate::console::Console;

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.tui {
        return tui::run_tui(&cli);
    }
    let console = Console::new(false, !cli.no_progress);
    let never = AtomicBool::new(false);
    let result = run::run_inference(&cli, &never, &|ev| console.emit(ev), !cli.no_progress);
    console.finish();
    result.map(|_| ())
}
