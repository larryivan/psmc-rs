//! The file picker of the TUI's form: an input file, or the directory of
//! the output.

use anyhow::{Context, Result};
use crossterm::event::KeyCode;
use std::fs;
use std::path::{Path, PathBuf};

use super::{App, FIELD_INPUT, FIELD_INPUT_FORMAT, FIELD_OUTPUT, parse_input_format};
use crate::cli::FormatArg;
use crate::dataset::is_supported_input_file;

/// What the file picker picks.
#[derive(Clone, Copy)]
pub(super) enum PickerMode {
    /// An input file of this format.
    Input(FormatArg),
    /// The directory of the output.
    OutputDir,
}

#[derive(Clone)]
pub(super) struct PickerEntry {
    pub(super) label: String,
    path: PathBuf,
    is_file: bool,
}

pub(super) struct FilePicker {
    pub(super) mode: PickerMode,
    pub(super) cwd: PathBuf,
    pub(super) entries: Vec<PickerEntry>,
    pub(super) selected: usize,
}

impl FilePicker {
    /// Go to `dir` and list it: itself, its parent, its directories and,
    /// when picking an input, its input files of the format.
    fn enter(&mut self, dir: PathBuf) -> Result<()> {
        let listing = fs::read_dir(&dir)
            .with_context(|| format!("failed to read directory {}", dir.display()))?;
        let entry = |label: String, path: PathBuf, is_file| PickerEntry {
            label,
            path,
            is_file,
        };
        let (mut dirs, mut files) = (Vec::new(), Vec::new());
        for item in listing {
            let path = item?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if path.is_dir() {
                dirs.push(entry(format!("{name}/"), path.clone(), false));
            } else if let PickerMode::Input(format) = self.mode
                && path.is_file()
                && is_supported_input_file(&path, format)
            {
                files.push(entry(name.to_string(), path.clone(), true));
            }
        }
        dirs.sort_by_key(|e| e.label.to_ascii_lowercase());
        files.sort_by_key(|e| e.label.to_ascii_lowercase());
        let here = entry("[use this directory]".to_string(), dir.clone(), false);
        let up = dir
            .parent()
            .map(|p| entry("../".to_string(), p.into(), false));
        self.entries = [here]
            .into_iter()
            .chain(up)
            .chain(dirs)
            .chain(files)
            .collect();
        self.selected = self.selected.min(self.entries.len() - 1);
        self.cwd = dir;
        Ok(())
    }
}

/// Where the picker for a field holding `raw_path` starts: that directory,
/// the directory of that file (or of a new file in an existing directory),
/// or the current directory.
fn start_dir(raw_path: &str) -> PathBuf {
    let p = Path::new(raw_path.trim());
    if p.is_dir() {
        return p.to_path_buf();
    }
    // A bare file name's parent is the empty path, which is not a directory
    // to list.
    match p.parent() {
        Some(dir) if dir.as_os_str().is_empty() && p.is_file() => PathBuf::from("."),
        Some(dir) if !dir.as_os_str().is_empty() && dir.exists() => dir.to_path_buf(),
        _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

impl App {
    /// Open the file picker for `field`: an input file of the format field's
    /// format for [`FIELD_INPUT`], the output's directory for
    /// [`FIELD_OUTPUT`]. It starts in the field's directory.
    pub(super) fn open_picker(&mut self, field: usize) -> Result<()> {
        let mode = match field {
            FIELD_INPUT => PickerMode::Input(parse_input_format(&self.fields[FIELD_INPUT_FORMAT])?),
            _ => PickerMode::OutputDir,
        };
        let mut picker = FilePicker {
            mode,
            cwd: PathBuf::new(),
            entries: Vec::new(),
            selected: 0,
        };
        picker.enter(start_dir(&self.fields[field]))?;
        self.picker = Some(picker);
        Ok(())
    }

    /// A key while the file picker is open.
    pub(super) fn picker_key(&mut self, code: KeyCode) -> Result<()> {
        let Some(picker) = self.picker.as_mut() else {
            return Ok(());
        };
        let n = picker.entries.len();
        let entry = picker.entries[picker.selected].clone();
        match code {
            KeyCode::Esc | KeyCode::Char('q') => self.picker = None,
            KeyCode::Up => picker.selected = (picker.selected + n - 1) % n,
            KeyCode::Down => picker.selected = (picker.selected + 1) % n,
            KeyCode::Right if !entry.is_file => picker.enter(entry.path)?,
            KeyCode::Backspace | KeyCode::Left => {
                if let Some(parent) = picker.cwd.parent() {
                    picker.enter(parent.to_path_buf())?;
                }
            }
            // An input needs a file, so a directory opens; the output takes
            // the directory.
            KeyCode::Enter => match (picker.mode, entry.is_file) {
                (PickerMode::Input(_), true) => {
                    self.fields[FIELD_INPUT] = entry.path.display().to_string();
                    self.picker = None;
                }
                (PickerMode::Input(_), false) => picker.enter(entry.path)?,
                (PickerMode::OutputDir, true) => {}
                (PickerMode::OutputDir, false) => {
                    let output = &mut self.fields[FIELD_OUTPUT];
                    let name = Path::new(output.as_str())
                        .file_name()
                        .map_or_else(|| "out.json".into(), |v| v.to_os_string());
                    *output = entry.path.join(name).display().to_string();
                    self.picker = None;
                }
            },
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::tests::{app, set};

    /// A file named without a directory opens the picker in the current
    /// directory (its parent is the empty path).
    #[test]
    fn the_picker_starts_in_the_directory_of_a_bare_file_name() {
        assert_eq!(start_dir("Cargo.toml"), PathBuf::from("."));
        let mut a = app(&[]);
        set(&mut a, FIELD_INPUT, "Cargo.toml");
        set(&mut a, FIELD_INPUT_FORMAT, "psmcfa");
        a.open_picker(FIELD_INPUT).unwrap();
    }

    /// The output-directory picker does not need the input's format.
    #[test]
    fn the_output_picker_opens_whatever_the_input_format_field_holds() {
        let mut a = app(&[]);
        set(&mut a, FIELD_INPUT_FORMAT, "not-a-format");
        assert!(a.open_picker(FIELD_INPUT).is_err());
        a.open_picker(FIELD_OUTPUT).unwrap();
    }
}
