//! Drawing the TUI: the setup form, the progress of a run, its results and
//! the file picker.

use ratatui::prelude::*;
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap,
};
use std::time::Instant;

use psmc_rs::qc::fmt::{count, counted};

use super::picker::{FilePicker, PickerMode};
use super::{
    App, FIELD_INPUT_FORMAT, FIELDS, Page, STAGES, StageSnapshot, build_cli_from_form, command_line,
};
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
    Style::new().bg(Color::Rgb(26, 48, 80)).white().bold()
}

/// `seconds` as MM:SS, or HH:MM:SS from an hour on.
fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// A stage's count, how long it has run (until `now` while it runs, until
/// its last update once complete or once the run is over), and the time
/// left at its rate so far; nothing before it starts. An M-step runs until
/// it converges, far below its cap (its total until its last event), so it
/// shows its iterations alone.
fn stage_label(stage: Stage, snap: &StageSnapshot, now: Option<Instant>) -> String {
    let Some((start, last)) = snap.started_at.zip(snap.updated_at) else {
        return String::new();
    };
    let iter = snap
        .iter
        .map_or(String::new(), |(i, n)| format!("EM {i}/{n} | "));
    let end = now.filter(|_| snap.done < snap.total).unwrap_or(last);
    let elapsed = clock((end - start).as_secs_f64());
    if stage == Stage::MStep {
        let iterations = counted(snap.done as usize, "iteration", "iterations");
        return format!("{iter}{iterations} | {elapsed} elapsed");
    }
    let rate = snap.done as f64 / (last - start).as_secs_f64();
    let eta = match snap.total.saturating_sub(snap.done) as f64 / rate {
        left if left.is_finite() => clock(left),
        _ => "--:--".to_string(),
    };
    let (done, total) = (count(snap.done), count(snap.total));
    format!("{iter}{done} / {total} | {elapsed} elapsed | ETA {eta}")
}

/// `line` cut into rows of at most `width` characters, after a space where
/// there is one.
fn rows(line: &str, width: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some((cut, _)) = rest.char_indices().nth(width) {
        let at = rest[..cut]
            .rfind(' ')
            .filter(|&i| i > 0)
            .map_or(cut, |i| i + 1);
        out.push(&rest[..at]);
        rest = &rest[at..];
    }
    out.push(rest);
    out
}

/// The end of the log, `app.scroll_back` rows before it (clamped to the
/// log's start). The rows are wrapped here rather than by the paragraph, so
/// that they can be counted.
fn render_log(f: &mut Frame, area: Rect, app: &mut App) {
    let width = usize::from(area.width.saturating_sub(2)).max(1);
    let height = usize::from(area.height.saturating_sub(2));
    // The last rows first, as many as the panel shows scrolled back.
    let mut last: Vec<&str> = Vec::new();
    for line in app.logs.iter().rev() {
        if last.len() >= height + app.scroll_back {
            break;
        }
        last.extend(rows(line, width).into_iter().rev());
    }
    app.scroll_back = app.scroll_back.min(last.len().saturating_sub(height));
    let shown: Vec<Line> = last
        .iter()
        .skip(app.scroll_back)
        .take(height)
        .rev()
        .map(|&row| Line::raw(row))
        .collect();
    f.render_widget(Paragraph::new(shown).block(panel("Log")), area);
}

/// A paragraph wrapped at its panel's width.
fn wrapped<'a>(text: impl Into<Text<'a>>, title: &'a str) -> Paragraph<'a> {
    Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .block(panel(title))
}

fn render_setup_page(f: &mut Frame, area: Rect, app: &mut App) {
    let [form, side] =
        Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).areas(area);
    let items: Vec<ListItem> = FIELDS
        .iter()
        .zip(&app.fields)
        .map(|((label, _), value)| {
            let value = match value.trim() {
                "" => "<empty>".dark_gray(),
                _ => value.as_str().white(),
            };
            ListItem::new(Line::from(vec![
                format!("{label:<11} ").light_blue().bold(),
                value,
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(panel("Setup"))
        .highlight_style(highlight())
        .highlight_symbol("▶ ");
    let mut state = ListState::default().with_selected(Some(app.selected));
    f.render_stateful_widget(list, form, &mut state);

    let [command, field, log] = Layout::vertical([
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Min(6),
    ])
    .areas(side);
    let text = match build_cli_from_form(app).map(|cli| command_line(&app.self_bin, &cli)) {
        Ok(line) => Text::raw(line),
        Err(e) => Text::raw(format!("Not ready: {e:#}")).yellow(),
    };
    f.render_widget(wrapped(text, "Command"), command);
    let value = app.fields[app.selected].as_str();
    let (label, hint) = FIELDS[app.selected];
    let about = vec![hint.gray().into(), Line::raw(value)];
    f.render_widget(wrapped(about, label), field);
    render_log(f, log, app);
}

fn render_run_page(f: &mut Frame, area: Rect, app: &mut App) {
    let gauges = STAGES.map(|_| Constraint::Length(3));
    let chunks = Layout::vertical(gauges.into_iter().chain([Constraint::Min(6)])).split(area);
    let now = app.running.as_ref().map(|_| Instant::now());
    for ((&(stage, title), snap), &chunk) in STAGES.iter().zip(&app.stages).zip(chunks.iter()) {
        let ratio = match snap.total {
            0 => 0.0,
            // An M-step's total is its iteration cap until it converges, so
            // its gauge fills only then.
            _ if stage == Stage::MStep && snap.done < snap.total => 0.0,
            total => (snap.done as f64 / total as f64).min(1.0),
        };
        let gauge = Gauge::default()
            .block(panel(title))
            .gauge_style(Style::new().cyan().bg(Color::Rgb(20, 30, 45)).bold())
            .label(stage_label(stage, snap, now))
            .ratio(ratio);
        f.render_widget(gauge, chunk);
    }
    render_log(f, chunks[STAGES.len()], app);
}

fn render_result_page(f: &mut Frame, area: Rect, app: &mut App) {
    let [results, log] = Layout::vertical([Constraint::Length(10), Constraint::Min(6)]).areas(area);
    let text = match app.results.as_slice() {
        [] => Text::raw("No results yet: fill in the form (F1) and press Enter to run."),
        lines => lines.iter().map(|l| Line::raw(l.as_str())).collect(),
    };
    f.render_widget(wrapped(text, "Results"), results);
    render_log(f, log, app);
}

/// The middle `percent_x` by `percent_y` of `r`.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let middle = |percent: u16| {
        let margin = Constraint::Percentage((100 - percent) / 2);
        [margin, Constraint::Percentage(percent), margin]
    };
    let [_, rows, _] = Layout::vertical(middle(percent_y)).areas(r);
    let [_, area, _] = Layout::horizontal(middle(percent_x)).areas(rows);
    area
}

fn render_picker(f: &mut Frame, picker: &FilePicker) {
    let popup = centered_rect(86, 82, f.area());
    let what = match picker.mode {
        PickerMode::Input(FormatArg::Auto) => "an input file".to_string(),
        PickerMode::Input(format) => format!("a {} file", format.label()),
        PickerMode::OutputDir => "the output directory".to_string(),
    };
    let items: Vec<ListItem> = picker
        .entries
        .iter()
        .map(|e| ListItem::new(e.label.as_str()))
        .collect();
    let list = List::new(items)
        .style(Style::new().white().bg(Color::Rgb(6, 10, 18)))
        .block(panel(format!("Pick {what}: {}", picker.cwd.display())))
        .highlight_symbol("▶ ")
        .highlight_style(highlight());
    let mut state = ListState::default().with_selected(Some(picker.selected));
    f.render_widget(Clear, popup);
    f.render_stateful_widget(list, popup, &mut state);
}

/// The keys of what is on screen, the most used first.
fn key_hints(app: &App) -> &'static str {
    if app.picker.is_some() {
        "↑/↓ move  Enter pick or open  → open  ←/Backspace up  Esc close"
    } else if app.running.is_some() {
        "x stop  ↑/↓/PgUp/PgDn scroll the log  F1/F2/F3 pages"
    } else if app.page != Page::Setup {
        "Enter run  ↑/↓/PgUp/PgDn scroll the log  F1 setup  q quit"
    } else if app.selected == FIELD_INPUT_FORMAT {
        "Tab/↑/↓ field  Space/←/→ change the format  Enter run  F2/F3 pages  Esc quit"
    } else {
        "Tab/↑/↓ field  type to edit  Ctrl+U clear  Ctrl+O input file  Ctrl+D output directory  \
         Enter run  F2/F3 pages  Esc quit"
    }
}

pub(super) fn render_ui(f: &mut Frame, app: &mut App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(10),
        Constraint::Length(4),
    ])
    .areas(f.area());
    let [tabs, status] =
        Layout::horizontal([Constraint::Length(42), Constraint::Min(20)]).areas(header);
    // The pages in the order of `Page`.
    let tabs_widget = Tabs::new([" F1 Setup ", " F2 Run ", " F3 Result "])
        .select(app.page as usize)
        .block(panel("psmc-rs"))
        .highlight_style(Style::new().cyan().bold())
        .gray();
    f.render_widget(tabs_widget, tabs);
    // Highlighted while a run is under way.
    let line = match app.running {
        Some(_) => app.status.as_str().yellow().bold(),
        None => app.status.as_str().into(),
    };
    f.render_widget(Paragraph::new(line).block(panel("Status")), status);

    match app.page {
        Page::Setup => render_setup_page(f, body, app),
        Page::Run => render_run_page(f, body, app),
        Page::Result => render_result_page(f, body, app),
    }
    f.render_widget(wrapped(key_hints(app), "Keys"), footer);
    if let Some(picker) = &app.picker {
        render_picker(f, picker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Nothing before a stage starts, an M-step's iterations alone, and the
    /// clock of a stage left unfinished stopped with the run.
    #[test]
    fn stage_labels_show_the_time_taken_and_left() {
        let t0 = Instant::now();
        let secs = |s| t0 + Duration::from_secs(s);
        let mut snap = StageSnapshot::default();
        assert_eq!(stage_label(Stage::EForward, &snap, Some(secs(9))), "");
        (snap.done, snap.total, snap.iter) = (20, 100, Some((2, 4)));
        (snap.started_at, snap.updated_at) = (Some(t0), Some(secs(10)));
        let label = |stage, now| stage_label(stage, &snap, now);
        assert_eq!(
            label(Stage::EForward, Some(secs(30))),
            "EM 2/4 | 20 / 100 | 00:30 elapsed | ETA 00:40"
        );
        assert_eq!(
            label(Stage::EForward, None),
            "EM 2/4 | 20 / 100 | 00:10 elapsed | ETA 00:40"
        );
        assert_eq!(
            label(Stage::MStep, None),
            "EM 2/4 | 20 iterations | 00:10 elapsed"
        );
        snap.done = 1;
        assert!(stage_label(Stage::MStep, &snap, None).contains("| 1 iteration |"));
    }

    #[test]
    fn log_lines_are_cut_into_rows_after_a_space() {
        assert_eq!(rows("", 4), [""]);
        assert_eq!(rows("abcd", 4), ["abcd"]);
        assert_eq!(rows("ab cd ef", 5), ["ab ", "cd ef"]);
        assert_eq!(rows("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    }
}
