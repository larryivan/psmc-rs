//! Drawing the TUI: the setup form, the progress of a run, its results and
//! the file picker.

use ratatui::prelude::*;
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap,
};

use super::{App, PickerMode, STAGE_ORDER, StageSnapshot, TuiPage, preview_cmdline, stage_idx};
use crate::cli::FormatArg;
use crate::run::Stage;

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

fn stage_eta_and_rate(stage: &StageSnapshot) -> (Option<f64>, Option<f64>) {
    let Some(start) = stage.started_at else {
        return (None, None);
    };
    if stage.done == 0 {
        return (None, None);
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rate = (stage.done as f64) / elapsed;
    if stage.total > stage.done && rate > 0.0 {
        let rem = (stage.total - stage.done) as f64;
        (Some(rem / rate), Some(rate))
    } else {
        (Some(0.0), Some(rate))
    }
}

fn fmt_seconds(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let ss = s % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{ss:02}")
    } else {
        format!("{m:02}:{ss:02}")
    }
}

fn fmt_rate(rate: f64, unit: &str) -> String {
    if unit == "B/s" {
        let mut v = rate;
        let mut suffix = "B/s";
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "KB/s";
        }
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "MB/s";
        }
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "GB/s";
        }
        return format!("{v:.1} {suffix}");
    }
    format!("{rate:.1} {unit}")
}

fn render_logs_panel(f: &mut Frame, area: Rect, app: &App, title: &str) {
    let visible = area.height.saturating_sub(2) as usize;
    let log_count = app.logs.len();
    let max_scroll = log_count.saturating_sub(visible) as u16;
    let scroll = if app.autoscroll {
        max_scroll
    } else {
        app.manual_scroll.min(max_scroll)
    };
    let log_widget = Paragraph::new(
        app.logs
            .iter()
            .map(|s| Line::raw(s.clone()))
            .collect::<Vec<_>>(),
    )
    .wrap(Wrap { trim: false })
    .scroll((scroll, 0))
    .block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(log_widget, area);
}

fn render_setup_page(f: &mut Frame, area: Rect, app: &App) {
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);

    let form_items = app
        .fields
        .iter()
        .map(|field| {
            let value_span = if field.value.trim().is_empty() {
                Span::styled("<empty>", Style::default().fg(Color::DarkGray))
            } else {
                Span::styled(field.value.clone(), Style::default().fg(Color::White))
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
        .collect::<Vec<_>>();

    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));
    let form = List::new(form_items)
        .block(
            Block::default()
                .title("Setup")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(26, 48, 80))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(form, body[0], &mut list_state);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
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
    .block(
        Block::default()
            .title("Command Preview")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(preview_widget, right[0]);

    let selected = &app.fields[app.selected];
    let selected_widget = Paragraph::new(format!(
        "{}\n{}",
        selected.label,
        if selected.value.trim().is_empty() {
            "<empty>".to_string()
        } else {
            selected.value.clone()
        }
    ))
    .wrap(Wrap { trim: false })
    .block(
        Block::default()
            .title("Selected Field (Full Value)")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(selected_widget, right[1]);

    render_logs_panel(f, right[2], app, "Recent Log");
}

fn render_run_page(f: &mut Frame, area: Rect, app: &App) {
    let mut constraints = Vec::<Constraint>::new();
    for _ in STAGE_ORDER {
        constraints.push(Constraint::Length(3));
    }
    constraints.push(Constraint::Min(8));
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, stage) in STAGE_ORDER.iter().enumerate() {
        let snap = &app.stages[stage_idx(*stage)];
        let ratio = if snap.total > 0 {
            (snap.done as f64 / snap.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (eta, rate) = stage_eta_and_rate(snap);
        let iter_txt = snap
            .iter
            .map(|(it, all)| format!(" | EM {it}/{all}"))
            .unwrap_or_default();
        let eta_txt = eta.map(fmt_seconds).unwrap_or_else(|| "--:--".to_string());
        let rate_txt = rate
            .map(|v| fmt_rate(v, stage_unit(*stage)))
            .unwrap_or_else(|| "--".to_string());
        let label = format!(
            "{}{iter_txt} | {}/{} | ETA {} | {}",
            stage_label(*stage),
            snap.done,
            snap.total,
            eta_txt,
            rate_txt
        );
        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(stage_label(*stage))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .gauge_style(
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(20, 30, 45))
                    .add_modifier(Modifier::BOLD),
            )
            .label(label)
            .ratio(ratio);
        f.render_widget(gauge, chunks[i]);
    }
    render_logs_panel(f, chunks[chunks.len() - 1], app, "Run Log");
}

fn render_result_page(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(8)])
        .split(area);
    let summary = if let Some(f) = &app.result {
        let r = &f.summary;
        format!(
            "Elapsed: {:.2}s\nOutput JSON: {}\nHTML report: {}\nTMRCA: {}\nBootstrap dir: {}\nLast EM loglike: {}",
            f.elapsed_sec,
            r.output_json.display(),
            r.html_report.display(),
            r.tmrca_out
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.bootstrap_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.final_loglike
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
        )
    } else {
        "No completed run yet.".to_string()
    };
    let summary_widget = Paragraph::new(summary).block(
        Block::default()
            .title("Result")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(summary_widget, chunks[0]);
    render_logs_panel(f, chunks[1], app, "Run Log");
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn render_picker(f: &mut Frame, app: &App) {
    let Some(picker) = app.picker.as_ref() else {
        return;
    };
    let popup = centered_rect(86, 82, f.area());
    f.render_widget(Clear, popup);
    f.render_widget(
        Block::default()
            .style(Style::default().bg(Color::Rgb(6, 10, 18)))
            .borders(Borders::NONE),
        popup,
    );
    let popup_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(2)])
        .split(popup);
    let items = picker
        .entries
        .iter()
        .map(|e| ListItem::new(e.label.clone()))
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    state.select(Some(picker.selected));
    let title = match picker.mode {
        PickerMode::InputPath => {
            let fmt = picker.input_format.unwrap_or(FormatArg::Auto);
            match fmt {
                FormatArg::Auto => "Input picker [any input]: Enter file=select, Enter dir=open",
                FormatArg::Psmcfa => "Input picker [psmcfa]: Enter file=select, Enter dir=open",
                FormatArg::Mhs => "Input picker [mhs]: Enter file=select, Enter dir=open",
                FormatArg::Vcf => "Input picker [vcf]: Enter file=select, Enter dir=open",
            }
        }
        PickerMode::OutputDir => "Output directory picker: Enter dir=select",
    };
    let list = List::new(items)
        .style(Style::default().fg(Color::White).bg(Color::Rgb(6, 10, 18)))
        .block(
            Block::default()
                .title(format!("{title} | cwd: {}", picker.cwd.display()))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_symbol("▶ ")
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(26, 48, 80))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
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
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(4),
        ])
        .split(f.area());

    let header = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(44), Constraint::Min(20)])
        .split(root[0]);

    let titles = vec![" Setup ", " Run ", " Result "];
    let selected_tab = match app.page {
        TuiPage::Setup => 0,
        TuiPage::Run => 1,
        TuiPage::Result => 2,
    };
    let tabs = Tabs::new(titles)
        .select(selected_tab)
        .block(
            Block::default()
                .title("PSMC-RS")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .style(Style::default().fg(Color::Gray));
    f.render_widget(tabs, header[0]);

    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            if app.running.is_some() {
                "RUNNING"
            } else {
                "READY"
            },
            Style::default()
                .fg(if app.running.is_some() {
                    Color::Yellow
                } else {
                    Color::Green
                })
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(&app.status),
    ]))
    .block(
        Block::default()
            .title("Status")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
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
        "Setup: Tab/Shift+Tab select field  type edit  Ctrl+O input picker  Ctrl+D output dir picker  Space toggle fmt  F5 run"
    };
    let secondary = "Pages: F1 Setup | F2 Run | F3 Result";
    let footer = Paragraph::new(vec![
        Line::raw(hint),
        Line::raw(format!(
            "{secondary} | Field hint: {}",
            app.fields[app.selected].hint
        )),
    ])
    .block(
        Block::default()
            .title("Help")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(footer, root[2]);

    render_picker(f, app);
}
