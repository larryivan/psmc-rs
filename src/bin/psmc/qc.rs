//! `psmc-rs qc`: the input QC report of a file, read as inference reads it.

use anyhow::Result;
use std::process::ExitCode;

use psmc_rs::qc::render_text;

use crate::cli::QcArgs;
use crate::console::{Console, print_err, print_out};
use crate::dataset::{describe, load_input, loader_warnings, qc_report};
use crate::output::{is_stdout, write_qc_json};
use crate::run::Event;

/// The exit status of `qc --strict` when the report has a warning.
pub const STRICT_FAILURE: u8 = 2;

/// Run `qc`: the report on stdout (on stderr when the JSON goes to stdout),
/// how the input was read on stderr.
pub fn run(args: &QcArgs) -> Result<ExitCode> {
    let console = Console::new(false, false);
    let sink = |ev| console.emit(ev);
    let data = load_input(&args.input_file, &args.input.to_options())?;
    for line in describe(&data) {
        sink(Event::Info(line));
    }
    let report = qc_report(
        &data,
        args.input.bin_size,
        args.bootstrap_block_size.unwrap_or(0),
        None,
    )?;
    for line in loader_warnings(&data, Some(&report)) {
        sink(Event::Warning(line));
    }
    let text = render_text(&report);
    let json_to_stdout = args.json.as_deref().is_some_and(is_stdout);
    if json_to_stdout {
        print_err(&text);
    } else {
        print_out(&text);
    }
    if let Some(path) = &args.json {
        write_qc_json(path, &report)?;
        if !json_to_stdout {
            sink(Event::Info(format!("QC report: {}", path.display())));
        }
    }
    if args.strict && report.has_warnings() {
        return Ok(ExitCode::from(STRICT_FAILURE));
    }
    Ok(ExitCode::SUCCESS)
}
