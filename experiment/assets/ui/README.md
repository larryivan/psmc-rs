UI screenshots for Supplementary Figure S3.

Expected files:

- `tui_run.png`: Terminal User Interface (TUI) screenshot during an active run.
- `html_report.png`: Browser screenshot of the generated interactive HTML report.

The plotting notebook reads them from this directory (repository-relative):

- `experiment/assets/ui/tui_run.png`
- `experiment/assets/ui/html_report.png`

If either file is missing, the notebook will render a placeholder panel in S3.
The screenshots show psmc-rs 0.1; retake them if the TUI or report changes.
