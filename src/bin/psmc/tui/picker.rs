//! The file picker of the TUI's form: an input file, or the directory of
//! the output.

use anyhow::{Context, Result};
use crossterm::event::KeyCode;
use std::fs;
use std::path::{Path, PathBuf};

use super::{App, FIELD_INPUT, FIELD_INPUT_FORMAT, FIELD_OUTPUT, parse_input_format_field};
use crate::cli::FormatArg;
use crate::dataset::is_supported_input_file;

/// What the file picker picks.
#[derive(Debug, Clone, Copy)]
pub(super) enum PickerMode {
    /// An input file of this format.
    Input(FormatArg),
    /// The directory of the output.
    OutputDir,
}

#[derive(Debug, Clone)]
pub(super) struct PickerEntry {
    pub(super) label: String,
    path: PathBuf,
    is_file: bool,
}

#[derive(Debug, Clone)]
pub(super) struct FilePicker {
    pub(super) mode: PickerMode,
    pub(super) cwd: PathBuf,
    pub(super) entries: Vec<PickerEntry>,
    pub(super) selected: usize,
}

impl FilePicker {
    fn open(mode: PickerMode, cwd: PathBuf) -> Result<Self> {
        let mut picker = Self {
            mode,
            cwd,
            entries: Vec::new(),
            selected: 0,
        };
        picker.refresh()?;
        Ok(picker)
    }

    /// Go to `dir` and list it.
    fn enter(&mut self, dir: PathBuf) -> Result<()> {
        self.cwd = dir;
        self.refresh()
    }

    /// List the current directory: itself, its parent, its directories and,
    /// when picking an input, its input files of the format.
    fn refresh(&mut self) -> Result<()> {
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.cwd)
            .with_context(|| format!("failed to read directory {}", self.cwd.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                dirs.push(PickerEntry {
                    label: format!("{name}/"),
                    path,
                    is_file: false,
                });
            } else if let PickerMode::Input(format) = self.mode
                && path.is_file()
                && is_supported_input_file(&path, format)
            {
                files.push(PickerEntry {
                    label: name,
                    path,
                    is_file: true,
                });
            }
        }
        dirs.sort_by_key(|e| e.label.to_ascii_lowercase());
        files.sort_by_key(|e| e.label.to_ascii_lowercase());
        let dir_entry = |label: &str, path: &Path| PickerEntry {
            label: label.to_string(),
            path: path.to_path_buf(),
            is_file: false,
        };
        let here = dir_entry("[use this directory]", &self.cwd);
        let up = self.cwd.parent().map(|parent| dir_entry("../", parent));
        self.entries = [here]
            .into_iter()
            .chain(up)
            .chain(dirs)
            .chain(files)
            .collect();
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        Ok(())
    }

    fn selected_entry(&self) -> Option<&PickerEntry> {
        self.entries.get(self.selected)
    }

    fn select_prev(&mut self) {
        self.selected = match self.selected {
            0 => self.entries.len().saturating_sub(1),
            i => i - 1,
        };
    }

    fn select_next(&mut self) {
        if !self.entries.is_empty() {
            self.selected = (self.selected + 1) % self.entries.len();
        }
    }
}

/// Where the picker for a field holding `raw_path` starts: that directory,
/// the directory of that file (or of a new file in an existing directory),
/// or the current directory.
fn start_dir(raw_path: &str) -> PathBuf {
    let p = Path::new(raw_path.trim());
    if !p.as_os_str().is_empty() {
        if p.is_dir() {
            return p.to_path_buf();
        }
        if p.is_file() {
            // A bare file name's parent is the empty path, which is not a directory to list.
            return p
                .parent()
                .filter(|d| !d.as_os_str().is_empty())
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        }
        if let Some(parent) = p.parent()
            && parent.exists()
        {
            return parent.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

impl App {
    /// Open the file picker for `field`: an input file of the format field's
    /// format for [`FIELD_INPUT`], the output's directory for
    /// [`FIELD_OUTPUT`]. It starts in the field's directory.
    pub(super) fn open_picker(&mut self, field: usize) -> Result<()> {
        let mode = match field {
            FIELD_INPUT => PickerMode::Input(parse_input_format_field(
                self.field_value(FIELD_INPUT_FORMAT),
            )?),
            _ => PickerMode::OutputDir,
        };
        let start = start_dir(self.field_value(field));
        self.picker = Some(FilePicker::open(mode, start)?);
        Ok(())
    }

    /// A key while the file picker is open.
    pub(super) fn picker_key(&mut self, code: KeyCode) -> Result<()> {
        let Some(picker) = self.picker.as_mut() else {
            return Ok(());
        };
        match code {
            KeyCode::Esc | KeyCode::Char('q') => self.picker = None,
            KeyCode::Up => picker.select_prev(),
            KeyCode::Down => picker.select_next(),
            KeyCode::Enter => self.pick()?,
            KeyCode::Right => {
                if let Some(entry) = picker.selected_entry().filter(|e| !e.is_file) {
                    picker.enter(entry.path.clone())?;
                }
            }
            KeyCode::Backspace | KeyCode::Left => {
                if let Some(parent) = picker.cwd.parent().map(Path::to_path_buf) {
                    picker.enter(parent)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Enter in the picker: take the selected input file, or the selected
    /// directory for the output; an input needs a file, so a directory opens.
    fn pick(&mut self) -> Result<()> {
        let Some(picker) = self.picker.as_mut() else {
            return Ok(());
        };
        let Some(entry) = picker.selected_entry().cloned() else {
            return Ok(());
        };
        match (picker.mode, entry.is_file) {
            (PickerMode::Input(_), true) => {
                self.fields[FIELD_INPUT].value = entry.path.display().to_string();
                self.picker = None;
            }
            (PickerMode::Input(_), false) => picker.enter(entry.path)?,
            (PickerMode::OutputDir, true) => {}
            (PickerMode::OutputDir, false) => {
                let output = &mut self.fields[FIELD_OUTPUT].value;
                let name = Path::new(output.as_str())
                    .file_name()
                    .map(|v| v.to_os_string())
                    .unwrap_or_else(|| "out.json".into());
                *output = entry.path.join(name).display().to_string();
                self.picker = None;
            }
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
