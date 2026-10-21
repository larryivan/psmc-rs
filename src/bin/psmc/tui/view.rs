//! Drawing the TUI: the setup form, the progress of a run, its results and
//! the file picker.

use ratatui::prelude::*;
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap,
};

use super::picker::PickerMode;
use super::{App, STAGE_ORDER, StageSnapshot, TuiPage, preview_cmdline};
use crate::cli::FormatArg;
use crate::run::Stage;

/// A titled panel with a rounded border.
fn panel<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
}

/// The style of the selected item of a list.
fn highlight() -> Style {
    Style::default()
        .bg(Color::Rgb(26, 48, 80))
        .fg(Color::White)
        .add_modifier(Modifier::BOLD)
}

fn stage_label(stage: Stage) -> &'static str {
    match stage {
        Stage::Io => "IO",
        Stage::EForward => "E-forward",
        Stage::EBackward => "E-backward",
        Stage::MStep => "M-step",
        Stage::Bootstrap => "Bootstrap",
    }
}

fn stage_unit(stage: Stage) -> &'static str {
    match stage {
        Stage::Io => "B/s",
        Stage::EForward | Stage::EBackward => "sites/s",
        Stage::MStep => "iter/s",
        Stage::Bootstrap => "rep/s",
    }
}

/// The seconds left and the rate of a stage that has made progress.
fn stage_eta_and_rate(stage: &StageSnapshot) -> Option<(f64, f64)> {
    let start = stage.started_at.filter(|_| stage.done > 0)?;
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rate = (stage.done as f64) / elapsed;
    if stage.total > stage.done && rate > 0.0 {
        let rem = (stage.total - stage.done) as f64;
        Some((rem / rate, rate))
    } else {
        Some((0.0, rate))
    }
}

fn fmt_seconds(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn fmt_rate(rate: f64, unit: &str) -> String {
    if unit != "B/s" {
        return format!("{rate:.1} {unit}");
    }
    let mut v = rate;
    let mut suffix = "B/s";
    for larger in ["KB/s", "MB/s", "GB/s"] {
        if v < 1024.0 {
            break;
        }
        v /= 1024.0;
        suffix = larger;
    }
    format!("{v:.1} {suffix}")
}

/// A field's value, or `<empty>`.
fn shown_value(value: &str) -> Option<&str> {
    (!value.trim().is_empty()).then_some(value)
}

fn render_logs_panel(f: &mut Frame, area: Rect, app: &App, title: &str) {
    let visible = area.height.saturating_sub(2) as usize;
    let max_scroll = app.logs.len().saturating_sub(visible) as u16;
    let scroll = if app.autoscroll {
        max_scroll
    } else {
        app.manual_scroll.min(max_scroll)
    };
    let lines: Vec<Line> = app.logs.iter().map(|s| Line::raw(s.clone())).collect();
    let log_widget = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(panel(title));
    f.render_widget(log_widget, area);
}

fn render_setup_page(f: &mut Frame, area: Rect, app: &App) {
    let body =
        Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).split(area);

    let form_items: Vec<ListItem> = app
        .fields
        .iter()
        .map(|field| {
            let value_span = match shown_value(&field.value) {
                Some(value) => Span::styled(value.to_string(), Style::default().fg(Color::White)),
                None => Span::styled("<empty>", Style::default().fg(Color::DarkGray)),
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<11}", field.label),
                    Style::default()
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                value_span,
            ]))
        })
        .collect();
    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));
    let form = List::new(form_items)
        .block(panel("Setup"))
        .highlight_style(highlight())
        .highlight_symbol("▶ ");
    f.render_stateful_widget(form, body[0], &mut list_state);

    let right = Layout::vertical([
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Min(6),
    ])
    .split(body[1]);

    let preview_widget = Paragraph::new(format!(
        "{}\n\nCtrl+O: pick input file\nCtrl+D: pick output directory\nF5: run",
        preview_cmdline(app)
    ))
    .wrap(Wrap { trim: false })
    .block(panel("Command Preview"));
    f.render_widget(preview_widget, right[0]);

    let selected = &app.fields[app.selected];
    let selected_widget = Paragraph::new(format!(
        "{}\n{}",
        selected.label,
        shown_value(&selected.value).unwrap_or("<empty>")
    ))
    .wrap(Wrap { trim: false })
    .block(panel("Selected Field (Full Value)"));
    f.render_widget(selected_widget, right[1]);

    render_logs_panel(f, right[2], app, "Recent Log");
}

fn render_run_page(f: &mut Frame, area: Rect, app: &App) {
    let gauges = STAGE_ORDER.map(|_| Constraint::Length(3));
    let chunks = Layout::vertical(gauges.into_iter().chain([Constraint::Min(8)])).split(area);

    for ((stage, snap), &chunk) in STAGE_ORDER.into_iter().zip(&app.stages).zip(chunks.iter()) {
        let ratio = if snap.total > 0 {
            (snap.done as f64 / snap.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let iter_txt = snap
            .iter
            .map(|(it, all)| format!(" | EM {it}/{all}"))
            .unwrap_or_default();
        let (eta_txt, rate_txt) = match stage_eta_and_rate(snap) {
            Some((eta, rate)) => (fmt_seconds(eta), fmt_rate(rate, stage_unit(stage))),
            None => ("--:--".to_string(), "--".to_string()),
        };
        let label = format!(
            "{}{iter_txt} | {}/{} | ETA {eta_txt} | {rate_txt}",
            stage_label(stage),
            snap.done,
            snap.total,
        );
        let gauge = Gauge::default()
            .block(panel(stage_label(stage)))
            .gauge_style(
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(20, 30, 45))
                    .add_modifier(Modifier::BOLD),
            )
            .label(label)
            .ratio(ratio);
        f.render_widget(gauge, chunk);
    }
    render_logs_panel(f, chunks[STAGE_ORDER.len()], app, "Run Log");
}

fn render_result_page(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::vertical([Constraint::Length(9), Constraint::Min(8)]).split(area);
    let summary = match &app.result {
        Some(done) => {
            let r = &done.summary;
            let or_dash = |v: Option<String>| v.unwrap_or_else(|| "-".to_string());
            format!(
                "Elapsed: {:.2}s\nOutput JSON: {}\nHTML report: {}\nTMRCA: {}\nBootstrap dir: {}\nLast EM loglike: {}",
                done.elapsed_sec,
                r.output_json.display(),
                r.html_report.display(),
                or_dash(r.tmrca_out.as_ref().map(|p| p.display().to_string())),
                or_dash(r.bootstrap_dir.as_ref().map(|p| p.display().to_string())),
                or_dash(r.final_loglike.map(|v| v.to_string())),
            )
        }
        None => "No completed run yet.".to_string(),
    };
    f.render_widget(Paragraph::new(summary).block(panel("Result")), chunks[0]);
    render_logs_panel(f, chunks[1], app, "Run Log");
}

/// The middle `percent_x` by `percent_y` of `r`.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let middle = |percent: u16| {
        let margin = Constraint::Percentage((100 - percent) / 2);
        [margin, Constraint::Percentage(percent), margin]
    };
    let vertical = Layout::vertical(middle(percent_y)).split(r);
    Layout::horizontal(middle(percent_x)).split(vertical[1])[1]
}

fn render_picker(f: &mut Frame, app: &App) {
    let Some(picker) = app.picker.as_ref() else {
        return;
    };
    let background = Color::Rgb(6, 10, 18);
    let popup = centered_rect(86, 82, f.area());
    f.render_widget(Clear, popup);
    f.render_widget(
        Block::default()
            .style(Style::default().bg(background))
            .borders(Borders::NONE),
        popup,
    );
    let popup_chunks = Layout::vertical([Constraint::Min(6), Constraint::Length(2)]).split(popup);
    let items: Vec<ListItem> = picker
        .entries
        .iter()
        .map(|e| ListItem::new(e.label.clone()))
        .collect();
    let mut state = ListState::default();
    state.select(Some(picker.selected));
    let title = match picker.mode {
        PickerMode::Input(format) => format!(
            "Input picker [{}]: Enter file=select, Enter dir=open",
            match format {
                FormatArg::Auto => "any input",
                format => format.name(),
            }
        ),
        PickerMode::OutputDir => "Output directory picker: Enter dir=select".to_string(),
    };
    let list = List::new(items)
        .style(Style::default().fg(Color::White).bg(background))
        .block(panel(format!("{title} | cwd: {}", picker.cwd.display())))
        .highlight_symbol("▶ ")
        .highlight_style(highlight());
    f.render_stateful_widget(list, popup_chunks[0], &mut state);

    let help = Paragraph::new(
        "Picker keys: Enter select/open | -> browse | <-/Backspace up | Esc/q close",
    )
    .style(Style::default().fg(Color::Gray))
    .block(
        Block::default()
            .borders(Borders::TOP)
            .border_type(BorderType::Plain),
    );
    f.render_widget(help, popup_chunks[1]);
}

pub(super) fn render_ui(f: &mut Frame, app: &App) {
    let root = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(4),
    ])
    .split(f.area());
    let header = Layout::horizontal([Constraint::Length(44), Constraint::Min(20)]).split(root[0]);

    // The pages in the order of `TuiPage`.
    let tabs = Tabs::new([" Setup ", " Run ", " Result "])
        .select(app.page as usize)
        .block(panel("PSMC-RS"))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .style(Style::default().fg(Color::Gray));
    f.render_widget(tabs, header[0]);

    let (state, color) = match app.running {
        Some(_) => ("RUNNING", Color::Yellow),
        None => ("READY", Color::Green),
    };
    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            state,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(&app.status),
    ]))
    .block(panel("Status"));
    f.render_widget(status, header[1]);

    match app.page {
        TuiPage::Setup => render_setup_page(f, root[1], app),
        TuiPage::Run => render_run_page(f, root[1], app),
        TuiPage::Result => render_result_page(f, root[1], app),
    }

    let hint = if app.picker.is_some() {
        "Picker: ↑/↓ move  Enter select/open  → browse dir  Backspace/← up  Esc/q close"
    } else if app.running.is_some() {
        "Run: F1/F2/F3 switch pages  PgUp/PgDn scroll logs  x stop  q quit"
    } else {
        "Setup: Tab/Shift+Tab select field  type edit  Ctrl+O input picker  Ctrl+D output dir picker  Space toggle fmt  F5 run  Esc quit"
    };
    let footer = Paragraph::new(vec![
        Line::raw(hint),
        Line::raw(format!(
            "Pages: F1 Setup | F2 Run | F3 Result | Field hint: {}",
            app.fields[app.selected].hint
        )),
    ])
    .block(panel("Help"));
    f.render_widget(footer, root[2]);

    render_picker(f, app);
}
