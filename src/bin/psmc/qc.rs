//! `psmc-rs qc`: the input QC report of a file, read as inference reads it.

use anyhow::Result;
use std::process::ExitCode;

use psmc_rs::qc::render_text;

use crate::cli::QcArgs;
use crate::console::{Console, print_err, print_out};
use crate::dataset::{default_qc_report, load_input, loader_warnings};
use crate::output::{check_destinations, is_stdout, write_qc_json};
use crate::run::{Event, announce_input};

/// The exit status of `qc --strict` when the report has a warning.
const STRICT_FAILURE: u8 = 2;

/// Run `qc`: the report on stdout (on stderr when the JSON goes to stdout),
/// how the input was read on stderr.
pub fn run(args: &QcArgs) -> Result<ExitCode> {
    let console = Console::new(false, false);
    let sink = |ev| console.emit(ev);
    let json_to_stdout = args.json.as_deref().is_some_and(is_stdout);
    // As inference and convert do: the JSON must be writable and must not
    // be the input, which is checked before it is read.
    let json_file = args.json.as_deref().filter(|_| !json_to_stdout);
    check_destinations(&args.input_file, &[("--json", json_file)], None)?;
    let data = load_input(&args.input_file, &args.input.to_options())?;
    announce_input(&data, &args.input, &sink);
    let trunk = args.bootstrap_block_size.unwrap_or(0);
    let report = default_qc_report(&data, args.input.bin_size, trunk)?;
    for line in loader_warnings(&data, Some(&report)) {
        sink(Event::Warning(line));
    }
    let print = if json_to_stdout { print_err } else { print_out };
    print(&render_text(&report));
    if let Some(path) = &args.json {
        write_qc_json(path, &report)?;
        if !json_to_stdout {
            sink(Event::Info(format!("QC report: {}", path.display())));
        }
    }
    let failed = args.strict && report.has_warnings();
    Ok(ExitCode::from(if failed { STRICT_FAILURE } else { 0 }))
}
