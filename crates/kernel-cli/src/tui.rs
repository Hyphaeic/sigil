//! Interactive TUI results viewer.
//!
//! This is a deliberately thin, read-only layer over [`SimulationResults`]:
//! the kernel remains a batch CLI tool, and the TUI is an opt-in viewer
//! (`simulate --tui`) for quick inspection without leaving the terminal.
//!
//! Views (tabs appear only when their data exists):
//! - **Summary**: configuration, channel info, eye metrics, PASS/FAIL
//! - **Eye (Statistical)**: worst-case eye envelope rails vs. phase
//! - **Eye (Bit-by-Bit)**: scope-style 2D density heatmap of the eye
//!   histogram (this data previously never left the orchestrator)
//! - **Pulse**: channel pulse response vs. time
//! - **Channel**: insertion loss vs. frequency with a Nyquist marker

use crate::config::SimulationConfig;
use crate::orchestrator::SimulationResults;
use anyhow::Result;
use lib_types::waveform::{EyeDiagram, Waveform};
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Tabs, Widget,
};
use ratatui::{DefaultTerminal, Frame};
use std::time::{Duration, Instant};

/// Animation frame interval (~30 fps while the eye is accumulating).
const TICK_RATE: Duration = Duration::from_millis(33);

/// Launch the interactive viewer over simulation results.
///
/// Requires an interactive terminal; errors out cleanly when stdout is not a
/// TTY (e.g. piped/scripted runs) so batch workflows are never disturbed.
pub fn run(results: &SimulationResults, config: &SimulationConfig) -> Result<()> {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        anyhow::bail!(
            "--tui requires an interactive terminal (stdout is not a TTY). \
             Results were still written to the output directory."
        );
    }

    let mut terminal = ratatui::init();
    let result = App::new(results, config).run(&mut terminal);
    ratatui::restore();
    result
}

/// Which view a tab shows.
#[derive(Clone, Copy, PartialEq)]
enum View {
    Summary,
    StatEye,
    BitEye,
    Pulse,
    Channel,
}

impl View {
    fn title(self) -> &'static str {
        match self {
            View::Summary => "Summary",
            View::StatEye => "Eye (Statistical)",
            View::BitEye => "Eye (Bit-by-Bit)",
            View::Pulse => "Pulse",
            View::Channel => "Channel",
        }
    }
}

/// Scope-style persistence replay: the bit-by-bit output waveform is fed
/// into a live histogram a batch of UIs per tick, so the eye "builds up"
/// on screen exactly the way the simulated data accumulated.
struct ReplayState {
    /// Histogram being accumulated (same dimensions as the final one).
    live: EyeDiagram,
    /// Next UI index of the output waveform to fold in.
    next_ui: usize,
    /// Total complete UIs available in the output waveform.
    total_ui: usize,
    /// UIs folded in per animation tick (speed; `+`/`-` adjust).
    uis_per_tick: usize,
    paused: bool,
}

impl ReplayState {
    fn done(&self) -> bool {
        self.next_ui >= self.total_ui
    }

    fn progress(&self) -> f64 {
        if self.total_ui == 0 {
            1.0
        } else {
            self.next_ui as f64 / self.total_ui as f64
        }
    }
}

struct App<'a> {
    results: &'a SimulationResults,
    config: &'a SimulationConfig,
    tabs: Vec<View>,
    selected: usize,
    // Pre-computed chart data (charts borrow slices at render time).
    stat_high: Vec<(f64, f64)>,
    stat_low: Vec<(f64, f64)>,
    pulse_pts: Vec<(f64, f64)>,
    pulse_cursors: Vec<(f64, f64)>,
    nyquist_marker: Vec<(f64, f64)>,
    replay: Option<ReplayState>,
}

impl<'a> App<'a> {
    fn new(results: &'a SimulationResults, config: &'a SimulationConfig) -> Self {
        let mut tabs = vec![View::Summary];

        let (stat_high, stat_low) = match &results.statistical_eye {
            Some(eye) if !eye.high.is_empty() => {
                tabs.push(View::StatEye);
                let n = eye.high.len() as f64;
                let high = eye.high.iter().enumerate().map(|(i, &v)| (i as f64 / n, v)).collect();
                let low = eye.low.iter().enumerate().map(|(i, &v)| (i as f64 / n, v)).collect();
                (high, low)
            }
            _ => (Vec::new(), Vec::new()),
        };

        // Live persistence replay needs both the final histogram (for the
        // binning parameters) and the raw output waveform to re-fold.
        let replay = match (&results.eye_diagram, &results.output_waveform) {
            (Some(eye), Some(wf)) if !wf.samples.is_empty() => {
                tabs.push(View::BitEye);
                let total_ui = wf.samples.len() / eye.samples_per_ui;
                Some(ReplayState {
                    live: EyeDiagram::new(eye.samples_per_ui, eye.voltage_bins, eye.voltage_range),
                    next_ui: 0,
                    total_ui,
                    // Default speed: full eye in ~10 s at 30 fps.
                    uis_per_tick: (total_ui / 300).max(25),
                    paused: false,
                })
            }
            (Some(_), _) => {
                tabs.push(View::BitEye);
                None // No waveform to replay; show the final histogram.
            }
            _ => None,
        };

        let (pulse_pts, pulse_cursors) = match &results.channel_pulse {
            Some(pulse) if !pulse.samples.is_empty() => {
                tabs.push(View::Pulse);
                pulse_chart_data(pulse, config.bit_time().0)
            }
            None | Some(_) => (Vec::new(), Vec::new()),
        };

        let mut nyquist_marker = Vec::new();
        if let Some(resp) = &results.channel_response {
            if !resp.points.is_empty() {
                tabs.push(View::Channel);
                let f_nyq_ghz = 1.0 / (2.0 * config.bit_time().0) * 1e-9;
                let y_min = resp.points.iter().map(|p| p.1).fold(f64::MAX, f64::min);
                nyquist_marker = vec![(f_nyq_ghz, y_min), (f_nyq_ghz, 0.0)];
            }
        }

        Self {
            results,
            config,
            tabs,
            selected: 0,
            stat_high,
            stat_low,
            pulse_pts,
            pulse_cursors,
            nyquist_marker,
            replay,
        }
    }

    /// Advance the persistence replay by one tick. Returns true if the
    /// display changed (a redraw is needed).
    fn tick_replay(&mut self) -> bool {
        let Some(replay) = &mut self.replay else {
            return false;
        };
        if replay.paused || replay.done() {
            return false;
        }
        let Some(wf) = &self.results.output_waveform else {
            return false;
        };

        let end = (replay.next_ui + replay.uis_per_tick).min(replay.total_ui);
        accumulate_ui_range(&mut replay.live, wf, replay.next_ui, end);
        replay.next_ui = end;
        true
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let mut needs_redraw = true;
        let mut last_tick = Instant::now();

        loop {
            if needs_redraw {
                terminal.draw(|frame| self.render(frame))?;
                needs_redraw = false;
            }

            // Wait for input up to the animation interval, then tick.
            let timeout = TICK_RATE.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                            self.selected = (self.selected + 1) % self.tabs.len();
                            needs_redraw = true;
                        }
                        KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                            self.selected =
                                (self.selected + self.tabs.len() - 1) % self.tabs.len();
                            needs_redraw = true;
                        }
                        KeyCode::Char(c @ '1'..='9') => {
                            let idx = (c as usize) - ('1' as usize);
                            if idx < self.tabs.len() {
                                self.selected = idx;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Char(' ') => {
                            if let Some(replay) = &mut self.replay {
                                replay.paused = !replay.paused;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Char('r') => {
                            if let Some(replay) = &mut self.replay {
                                let e = &replay.live;
                                replay.live = EyeDiagram::new(
                                    e.samples_per_ui,
                                    e.voltage_bins,
                                    e.voltage_range,
                                );
                                replay.next_ui = 0;
                                replay.paused = false;
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            if let Some(replay) = &mut self.replay {
                                replay.uis_per_tick = (replay.uis_per_tick * 2).min(1_000_000);
                                needs_redraw = true;
                            }
                        }
                        KeyCode::Char('-') => {
                            if let Some(replay) = &mut self.replay {
                                replay.uis_per_tick = (replay.uis_per_tick / 2).max(5);
                                needs_redraw = true;
                            }
                        }
                        _ => {}
                    },
                    Event::Resize(_, _) => needs_redraw = true,
                    _ => {}
                }
            }

            if last_tick.elapsed() >= TICK_RATE {
                if self.tick_replay() {
                    needs_redraw = true;
                }
                last_tick = Instant::now();
            }
        }
    }

    fn render(&self, frame: &mut Frame) {
        let [tab_area, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let titles: Vec<String> = self
            .tabs
            .iter()
            .enumerate()
            .map(|(i, v)| format!("{}:{}", i + 1, v.title()))
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.selected)
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        frame.render_widget(tabs, tab_area);

        match self.tabs[self.selected] {
            View::Summary => self.render_summary(frame, body),
            View::StatEye => self.render_stat_eye(frame, body),
            View::BitEye => self.render_bit_eye(frame, body),
            View::Pulse => self.render_pulse(frame, body),
            View::Channel => self.render_channel(frame, body),
        }

        let key_style = Style::default().fg(Color::Black).bg(Color::DarkGray);
        let mut hints = vec![
            Span::styled(" q ", key_style),
            Span::raw(" quit  "),
            Span::styled(" Tab/←→ ", key_style),
            Span::raw(" switch  "),
            Span::styled(" 1-9 ", key_style),
            Span::raw(" jump "),
        ];
        if self.tabs[self.selected] == View::BitEye && self.replay.is_some() {
            hints.extend([
                Span::raw(" "),
                Span::styled(" space ", key_style),
                Span::raw(" pause  "),
                Span::styled(" r ", key_style),
                Span::raw(" replay  "),
                Span::styled(" +/- ", key_style),
                Span::raw(" speed "),
            ]);
        }
        frame.render_widget(Paragraph::new(Line::from(hints)), footer);
    }

    /// Pass/fail verdict using the same criterion as `output.rs`.
    fn verdict(&self) -> Option<(bool, f64, f64)> {
        self.results
            .eye_metrics
            .map(|m| (m.height > 0.0 && m.width_ui > 0.3, m.height, m.width_ui))
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let gen = format!("{:?}", self.config.pcie_gen);
        let rate_gbps = self.config.data_rate().0 / 1e9;
        let ui_ps = self.config.bit_time().as_ps();
        let f_nyq_ghz = 1.0 / (2.0 * self.config.bit_time().0) * 1e-9;

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("Simulation  ", Style::default().fg(Color::DarkGray)),
                Span::raw(self.config.name.clone()),
            ]),
            Line::from(vec![
                Span::styled("Link        ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "PCIe {} — {:.0} GT/s, UI = {:.2} ps, Nyquist = {:.2} GHz",
                    gen, rate_gbps, ui_ps, f_nyq_ghz
                )),
            ]),
            Line::from(vec![
                Span::styled("Mode        ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "{:?}  (PRBS-{}, {} bits, {} samples/UI)",
                    self.config.simulation.mode,
                    self.config.simulation.prbs_order,
                    self.config.simulation.num_bits,
                    self.config.simulation.samples_per_ui
                )),
            ]),
            Line::from(vec![
                Span::styled("Channel     ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!("{}", self.config.channel.touchstone.display())),
            ]),
        ];

        if let Some(resp) = &self.results.channel_response {
            if let (Some(first), Some(last)) = (resp.points.first(), resp.points.last()) {
                // Loss at the point closest to Nyquist.
                let il_nyq = resp
                    .points
                    .iter()
                    .min_by(|a, b| {
                        (a.0 - f_nyq_ghz)
                            .abs()
                            .partial_cmp(&(b.0 - f_nyq_ghz).abs())
                            .unwrap()
                    })
                    .map(|p| p.1);
                let mut text = format!(
                    "{}: {:.2}–{:.2} GHz, {} points",
                    resp.label,
                    first.0,
                    last.0,
                    resp.points.len()
                );
                if let Some(il) = il_nyq {
                    text.push_str(&format!(", {:.2} dB near Nyquist", il));
                }
                lines.push(Line::from(vec![
                    Span::styled("Response    ", Style::default().fg(Color::DarkGray)),
                    Span::raw(text),
                ]));
            }
        }

        lines.push(Line::raw(""));

        if let Some((pass, height, width)) = self.verdict() {
            let health = |ok: bool| {
                if ok {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Red)
                }
            };
            lines.push(Line::from(vec![
                Span::styled("Eye height  ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.4} V", height), health(height > 0.0)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Eye width   ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.2} UI", width), health(width > 0.3)),
            ]));
            if let Some(eye) = &self.results.eye_diagram {
                lines.push(Line::from(vec![
                    Span::styled("UI count    ", Style::default().fg(Color::DarkGray)),
                    Span::raw(format!("{}", eye.ui_count)),
                ]));
            }
            lines.push(Line::raw(""));
            lines.push(Line::from(if pass {
                Span::styled(
                    " PASS — eye is open ",
                    Style::default().fg(Color::Black).bg(Color::Green).bold(),
                )
            } else {
                Span::styled(
                    " FAIL — eye is closed or marginal ",
                    Style::default().fg(Color::White).bg(Color::Red).bold(),
                )
            }));
        } else {
            lines.push(Line::raw("No eye metrics computed."));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Simulation Summary ");
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_stat_eye(&self, frame: &mut Frame, area: Rect) {
        let bound = self
            .stat_high
            .iter()
            .chain(self.stat_low.iter())
            .map(|p| p.1.abs())
            .fold(0.0, f64::max)
            .max(1e-6)
            * 1.15;

        let datasets = vec![
            Dataset::default()
                .name("high rail")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&self.stat_high),
            Dataset::default()
                .name("low rail")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&self.stat_low),
        ];

        let title = match self.verdict() {
            Some((_, h, w)) => format!(
                " Statistical Eye Envelope — height {:.3} V, width {:.2} UI ",
                h, w
            ),
            None => " Statistical Eye Envelope ".to_string(),
        };

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_axis(
                Axis::default()
                    .title("phase (UI)")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, 1.0])
                    .labels(["0", "0.5", "1"]),
            )
            .y_axis(
                Axis::default()
                    .title("V")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([-bound, bound])
                    .labels([
                        format!("{:+.2}", -bound),
                        "0".to_string(),
                        format!("{:+.2}", bound),
                    ]),
            );
        frame.render_widget(chart, area);
    }

    fn render_bit_eye(&self, frame: &mut Frame, area: Rect) {
        // Prefer the live replay histogram; fall back to the final one.
        let (eye, replay) = match &self.replay {
            Some(replay) => (&replay.live, Some(replay)),
            None => match &self.results.eye_diagram {
                Some(eye) => (eye, None),
                None => return,
            },
        };

        let title = match replay {
            Some(r) if !r.done() => format!(
                " Bit-by-Bit Eye — accumulating {} / {} UI ({}{}) ",
                r.next_ui,
                r.total_ui,
                if r.paused { "paused, " } else { "" },
                format_args!("{} UI/frame", r.uis_per_tick),
            ),
            _ => format!(
                " Bit-by-Bit Eye — {} UI accumulated, window {:+.1}..{:+.1} V ",
                eye.ui_count, eye.voltage_range.0, eye.voltage_range.1
            ),
        };

        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        match replay {
            Some(r) if !r.done() => {
                // Heatmap above, persistence progress gauge below.
                let [plot, gauge_area] =
                    Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(inner);
                frame.render_widget(EyeHeatmap { eye }, plot);

                let label = if r.paused { "paused (space resumes)" } else { "persistence" };
                let gauge = Gauge::default()
                    .ratio(r.progress())
                    .label(format!("{} {:.0}%", label, r.progress() * 100.0))
                    .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black));
                frame.render_widget(gauge, gauge_area);
            }
            _ => frame.render_widget(EyeHeatmap { eye }, inner),
        }
    }

    fn render_pulse(&self, frame: &mut Frame, area: Rect) {
        let (mut y_min, mut y_max) = (f64::MAX, f64::MIN);
        let (mut x_min, mut x_max) = (f64::MAX, f64::MIN);
        for &(x, y) in &self.pulse_pts {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            x_min = x_min.min(x);
            x_max = x_max.max(x);
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            return;
        }
        let pad = ((y_max - y_min) * 0.1).max(1e-6);

        let datasets = vec![
            Dataset::default()
                .name("pulse")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&self.pulse_pts),
            Dataset::default()
                .name("UI cursors")
                .marker(Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(Color::Yellow))
                .data(&self.pulse_cursors),
        ];

        // Cursor 0 is the main cursor; cursor 1 the first post-cursor (ISI).
        let title = match (self.pulse_cursors.first(), self.pulse_cursors.get(1)) {
            (Some(main), Some(post1)) => format!(
                " Channel Pulse Response — main {:.3} V, post-1 {:+.3} V ({:+.0}%), 1 UI = {:.2} ps ",
                main.1,
                post1.1,
                100.0 * post1.1 / main.1.max(1e-12),
                self.config.bit_time().as_ps()
            ),
            _ => format!(
                " Channel Pulse Response — peak {:.3} V, 1 UI = {:.2} ps ",
                y_max.abs().max(y_min.abs()),
                self.config.bit_time().as_ps()
            ),
        };

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(title))
            .x_axis(
                Axis::default()
                    .title("ns")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max])
                    .labels([
                        format!("{:.1}", x_min),
                        format!("{:.1}", (x_min + x_max) / 2.0),
                        format!("{:.1}", x_max),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("V")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_min - pad, y_max + pad])
                    .labels([
                        format!("{:+.2}", y_min - pad),
                        format!("{:+.2}", (y_min + y_max) / 2.0),
                        format!("{:+.2}", y_max + pad),
                    ]),
            );
        frame.render_widget(chart, area);
    }

    fn render_channel(&self, frame: &mut Frame, area: Rect) {
        let Some(resp) = &self.results.channel_response else {
            return;
        };
        let x_max = resp.points.last().map(|p| p.0).unwrap_or(1.0);
        let y_min = resp
            .points
            .iter()
            .map(|p| p.1)
            .fold(f64::MAX, f64::min)
            .min(-1.0);

        let datasets = vec![
            Dataset::default()
                .name(resp.label.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&resp.points),
            Dataset::default()
                .name("Nyquist")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&self.nyquist_marker),
        ];

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(format!(
                " Channel Insertion Loss |{}| ",
                resp.label
            )))
            .x_axis(
                Axis::default()
                    .title("GHz")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, x_max])
                    .labels([
                        "0".to_string(),
                        format!("{:.1}", x_max / 2.0),
                        format!("{:.1}", x_max),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("dB")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_min * 1.05, 1.0])
                    .labels([
                        format!("{:.0}", y_min * 1.05),
                        format!("{:.0}", y_min * 1.05 / 2.0),
                        "0".to_string(),
                    ]),
            );
        frame.render_widget(chart, area);
    }
}

/// Scope-style eye density heatmap.
///
/// Renders the [`EyeDiagram`] histogram with half-block characters (two
/// vertical pixels per terminal cell) and a logarithmic color ramp, with a
/// small voltage gutter on the left.
struct EyeHeatmap<'a> {
    eye: &'a EyeDiagram,
}

const GUTTER: u16 = 9;

impl Widget for EyeHeatmap<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width <= GUTTER + 2 || area.height < 2 {
            return;
        }
        let plot_x = area.x + GUTTER;
        let plot_w = (area.width - GUTTER) as usize;
        let plot_h = area.height as usize;
        let px_rows = plot_h * 2;

        let max_count = self
            .eye
            .bins
            .iter()
            .flat_map(|col| col.iter())
            .copied()
            .max()
            .unwrap_or(0);
        if max_count == 0 {
            buf.set_string(
                plot_x,
                area.y,
                "(eye histogram is empty)",
                Style::default().fg(Color::DarkGray),
            );
            return;
        }
        let ln_max = (1.0 + max_count as f64).ln();

        let phases = self.eye.samples_per_ui;
        let vbins = self.eye.voltage_bins;

        // Max density in the histogram rectangle covered by one screen pixel.
        // Max (not mean) keeps sparse traces visible at coarse resolutions.
        let count_at = |col: usize, prow: usize| -> u64 {
            let p0 = col * phases / plot_w;
            let p1 = (((col + 1) * phases).div_ceil(plot_w)).clamp(p0 + 1, phases);
            let r0 = prow * vbins / px_rows;
            let r1 = (((prow + 1) * vbins).div_ceil(px_rows)).clamp(r0 + 1, vbins);
            let mut m = 0u64;
            for p in p0..p1 {
                for r in r0..r1 {
                    // Pixel row 0 is the top of the plot = highest voltage bin.
                    let vbin = vbins - 1 - r;
                    m = m.max(self.eye.bins[p][vbin]);
                }
            }
            m
        };

        for row in 0..plot_h {
            for col in 0..plot_w {
                let top = count_at(col, row * 2);
                let bot = count_at(col, row * 2 + 1);
                if top == 0 && bot == 0 {
                    continue;
                }
                let cell = &mut buf[(plot_x + col as u16, area.y + row as u16)];
                match (top, bot) {
                    (0, b) => {
                        // Lower half block: only the bottom pixel is lit.
                        cell.set_symbol("▄");
                        cell.set_fg(density_color(b, ln_max));
                        cell.set_bg(Color::Reset);
                    }
                    (t, 0) => {
                        cell.set_symbol("▀");
                        cell.set_fg(density_color(t, ln_max));
                        cell.set_bg(Color::Reset);
                    }
                    (t, b) => {
                        cell.set_symbol("▀");
                        cell.set_fg(density_color(t, ln_max));
                        cell.set_bg(density_color(b, ln_max));
                    }
                }
            }
        }

        // Voltage gutter labels (top / mid / bottom) and UI extents.
        let (v_min, v_max) = self.eye.voltage_range;
        let label_style = Style::default().fg(Color::DarkGray);
        buf.set_string(area.x, area.y, format!("{:>+7.2} V", v_max), label_style);
        buf.set_string(
            area.x,
            area.y + (plot_h as u16) / 2,
            format!("{:>+7.2} V", (v_max + v_min) / 2.0),
            label_style,
        );
        buf.set_string(
            area.x,
            area.y + plot_h as u16 - 1,
            format!("{:>+7.2} V", v_min),
            label_style,
        );
    }
}

/// Logarithmic density → color ramp (deep blue → cyan → green → yellow → red).
fn density_color(count: u64, ln_max: f64) -> Color {
    let t = ((1.0 + count as f64).ln() / ln_max).clamp(0.0, 1.0);

    // Piecewise-linear interpolation across the ramp stops.
    const STOPS: [(f64, (u8, u8, u8)); 5] = [
        (0.00, (10, 25, 90)),
        (0.35, (0, 160, 255)),
        (0.60, (60, 220, 100)),
        (0.82, (255, 215, 0)),
        (1.00, (255, 60, 40)),
    ];

    let mut rgb = STOPS[STOPS.len() - 1].1;
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
            rgb = (lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
            break;
        }
    }
    Color::Rgb(rgb.0, rgb.1, rgb.2)
}

/// Build the pulse-response chart data: a decimated trace auto-zoomed to the
/// region that actually contains energy, plus UI-spaced cursor samples.
///
/// The raw pulse record spans the whole FFT window (hundreds of ns) while the
/// pulse itself occupies a few UI — plotting the full record collapses it
/// into a single invisible column. We window to where |v| exceeds 0.5% of the
/// peak (with a few UI of padding) before decimating.
///
/// Cursors are emitted with the main cursor first, then post-cursors in
/// order (the values a DFE would cancel), then pre-cursors.
fn pulse_chart_data(pulse: &Waveform, bit_time_s: f64) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let samples = &pulse.samples;
    let dt = pulse.dt.0;
    let spb = ((bit_time_s / dt).round() as usize).max(1);

    let (peak_idx, peak_abs) = samples
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .fold((0, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });

    if peak_abs <= 0.0 {
        let pts = decimate_minmax(samples, pulse.t_start.0, dt, 600);
        return (pts, Vec::new());
    }

    let threshold = 0.005 * peak_abs;
    let first = samples.iter().position(|v| v.abs() >= threshold).unwrap_or(0);
    let last = samples
        .iter()
        .rposition(|v| v.abs() >= threshold)
        .unwrap_or(samples.len() - 1);

    let pad = ((last - first) / 8).max(2 * spb);
    let w0 = first.saturating_sub(pad);
    let w1 = (last + pad + 1).min(samples.len());

    let pts = decimate_minmax(
        &samples[w0..w1],
        pulse.t_start.0 + w0 as f64 * dt,
        dt,
        600,
    );

    let t_ns_at = |idx: usize| (pulse.t_start.0 + idx as f64 * dt) * 1e9;
    let mut cursors = vec![(t_ns_at(peak_idx), samples[peak_idx])];
    for k in 1..=10usize {
        let idx = peak_idx + k * spb;
        if idx >= w1 {
            break;
        }
        cursors.push((t_ns_at(idx), samples[idx]));
    }
    for k in 1..=3usize {
        match peak_idx.checked_sub(k * spb) {
            Some(idx) if idx >= w0 => cursors.push((t_ns_at(idx), samples[idx])),
            _ => break,
        }
    }

    (pts, cursors)
}

/// Fold UIs `[ui_start, ui_end)` of a waveform into an eye histogram.
///
/// Incremental equivalent of [`EyeDiagram::accumulate`] used by the
/// persistence replay; identical binning math.
fn accumulate_ui_range(eye: &mut EyeDiagram, waveform: &Waveform, ui_start: usize, ui_end: usize) {
    let spui = eye.samples_per_ui;
    let max_ui = waveform.samples.len() / spui;
    let (v_min, v_max) = eye.voltage_range;
    let span = v_max - v_min;

    for ui in ui_start..ui_end.min(max_ui) {
        for phase in 0..spui {
            let voltage = waveform.samples[ui * spui + phase];
            let v_norm = (voltage - v_min) / span;
            let v_bin = (v_norm * eye.voltage_bins as f64)
                .floor()
                .max(0.0)
                .min((eye.voltage_bins - 1) as f64) as usize;
            eye.bins[phase][v_bin] += 1;
        }
        eye.ui_count += 1;
    }
}

/// Decimate a waveform for plotting, preserving the min/max envelope so peaks
/// survive aggressive downsampling. X values are emitted in nanoseconds.
fn decimate_minmax(samples: &[f64], t_start: f64, dt: f64, target_buckets: usize) -> Vec<(f64, f64)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let bucket = samples.len().div_ceil(target_buckets).max(1);
    let mut points = Vec::with_capacity(2 * target_buckets);

    for (b, chunk) in samples.chunks(bucket).enumerate() {
        let t_ns = (t_start + (b * bucket) as f64 * dt) * 1e9;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for &v in chunk {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        points.push((t_ns, lo));
        if hi > lo {
            points.push((t_ns, hi));
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChannelConfig, OutputConfig, SimulationParams};
    use crate::orchestrator::ChannelResponse;
    use lib_dsp::eye::EyeMetrics;
    use lib_types::units::Seconds;
    use lib_types::waveform::{StatisticalEye, Waveform};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn synthetic_config() -> SimulationConfig {
        SimulationConfig {
            name: "tui-test".to_string(),
            pcie_gen: Default::default(),
            channel: ChannelConfig {
                touchstone: "test.s2p".into(),
                mode: Default::default(),
                input_port: None,
                output_port: None,
            },
            tx: None,
            rx: None,
            simulation: SimulationParams::default(),
            output: OutputConfig::default(),
        }
    }

    fn synthetic_results() -> SimulationResults {
        // Statistical eye: open in the middle, closed at the crossing.
        let n = 64;
        let mut stat = StatisticalEye::new(n);
        for i in 0..n {
            let x = i as f64 / n as f64;
            let opening = (std::f64::consts::PI * x).sin() - 0.2;
            stat.high[i] = opening.max(-0.1);
            stat.low[i] = -opening.max(-0.1);
        }

        // Bit-by-bit eye histogram from a square-ish waveform; the same
        // waveform doubles as the output waveform so the replay has data.
        let spui = 16;
        let mut eye = EyeDiagram::new(spui, 32, (-1.5, 1.5));
        let samples: Vec<f64> = (0..spui * 50)
            .map(|i| if (i / spui) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let wf = Waveform::new(samples, Seconds::from_ps(1.0), Seconds::ZERO);
        eye.accumulate(&wf, spui);
        let output_waveform = wf.clone();

        // Pulse response: a raised-cosine-ish bump.
        let pulse_samples: Vec<f64> = (0..512)
            .map(|i| {
                let x = (i as f64 - 128.0) / 24.0;
                0.7 * (-x * x / 2.0).exp()
            })
            .collect();
        let pulse = Waveform::new(pulse_samples, Seconds::from_ps(25.0), Seconds::ZERO);

        // Insertion loss: smooth roll-off to -20 dB at 20 GHz.
        let points: Vec<(f64, f64)> = (1..=100)
            .map(|i| {
                let f = i as f64 * 0.2;
                (f, -f)
            })
            .collect();

        SimulationResults {
            statistical_eye: Some(stat),
            eye_metrics: Some(EyeMetrics {
                height: 0.75,
                width_ui: 0.67,
                jitter_rms: 0.0,
                snr: 0.0,
                ui_count: 50,
            }),
            eye_diagram: Some(eye),
            channel_pulse: Some(pulse),
            output_waveform: Some(output_waveform),
            channel_response: Some(ChannelResponse {
                label: "S21".to_string(),
                points,
            }),
            training_result: None,
        }
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn test_all_tabs_render_without_panic() {
        let results = synthetic_results();
        let config = synthetic_config();
        let mut app = App::new(&results, &config);

        // All five views should be present with full synthetic data.
        assert_eq!(app.tabs.len(), 5);

        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        for i in 0..app.tabs.len() {
            app.selected = i;
            terminal.draw(|f| app.render(f)).unwrap();
        }
    }

    #[test]
    fn test_summary_shows_verdict_and_metrics() {
        let results = synthetic_results();
        let config = synthetic_config();
        let app = App::new(&results, &config);

        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();

        let text = buffer_text(&terminal);
        assert!(text.contains("PASS"), "summary should show verdict");
        assert!(text.contains("0.7500"), "summary should show eye height");
        assert!(text.contains("tui-test"), "summary should show sim name");
    }

    #[test]
    fn test_heatmap_renders_density_cells() {
        let results = synthetic_results();
        let config = synthetic_config();
        let mut app = App::new(&results, &config);
        app.selected = app.tabs.iter().position(|v| *v == View::BitEye).unwrap();

        // Run the persistence replay to completion, then render.
        while app.replay.as_ref().is_some_and(|r| !r.done()) {
            assert!(app.tick_replay(), "replay should advance until done");
        }

        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();

        let text = buffer_text(&terminal);
        assert!(
            text.contains('▀') || text.contains('▄'),
            "heatmap should paint half-block density cells"
        );
        assert!(text.contains("UI accumulated"), "heatmap title missing");
    }

    #[test]
    fn test_replay_matches_full_accumulate() {
        // Incremental accumulation in arbitrary batches must reproduce the
        // exact histogram of EyeDiagram::accumulate.
        let spui = 16;
        let samples: Vec<f64> = (0..spui * 37)
            .map(|i| ((i as f64) * 0.37).sin() * 1.2)
            .collect();
        let wf = Waveform::new(samples, Seconds::from_ps(1.0), Seconds::ZERO);

        let mut reference = EyeDiagram::new(spui, 32, (-1.5, 1.5));
        reference.accumulate(&wf, spui);

        let mut incremental = EyeDiagram::new(spui, 32, (-1.5, 1.5));
        let total = wf.samples.len() / spui;
        let mut ui = 0;
        for batch in [1usize, 7, 3, 100] {
            let end = (ui + batch).min(total);
            accumulate_ui_range(&mut incremental, &wf, ui, end);
            ui = end;
        }

        assert_eq!(incremental.ui_count, reference.ui_count);
        assert_eq!(incremental.bins, reference.bins);
    }

    #[test]
    fn test_replay_renders_progress_gauge_midway() {
        let results = synthetic_results();
        let config = synthetic_config();
        let mut app = App::new(&results, &config);
        app.selected = app.tabs.iter().position(|v| *v == View::BitEye).unwrap();

        // One tick: partially accumulated → progress gauge visible.
        assert!(app.tick_replay());
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();

        let text = buffer_text(&terminal);
        assert!(text.contains("accumulating"), "live title missing");
        assert!(text.contains("%"), "progress gauge missing");
    }

    #[test]
    fn test_pulse_zoom_focuses_on_pulse_energy() {
        // A short bump inside a long mostly-zero record (the real shape of
        // sparam-derived pulses) must be zoomed to, not plotted full-width.
        let dt = 1e-12;
        let bit_time = 40e-12; // 40 samples/UI
        let mut samples = vec![0.0; 100_000];
        for i in 0..40 {
            samples[5_000 + i] = 0.7;
        }
        let pulse = Waveform::new(samples, Seconds(dt), Seconds::ZERO);

        let (pts, cursors) = pulse_chart_data(&pulse, bit_time);

        let x_min = pts.iter().map(|p| p.0).fold(f64::MAX, f64::min);
        let x_max = pts.iter().map(|p| p.0).fold(f64::MIN, f64::max);
        let full_ns = 100_000.0 * dt * 1e9;
        assert!(
            (x_max - x_min) < full_ns / 10.0,
            "window {:.1}..{:.1} ns should be a small slice of {:.1} ns",
            x_min,
            x_max,
            full_ns
        );

        // Main cursor first, at the pulse amplitude; post-cursor 1 near zero.
        assert!((cursors[0].1 - 0.7).abs() < 1e-9, "main cursor wrong: {:?}", cursors[0]);
        assert!(cursors[1].1.abs() < 1e-9, "post-1 should be ~0: {:?}", cursors[1]);
    }

    #[test]
    fn test_tabs_follow_available_data() {
        // Statistical-only results must not offer bit-by-bit/pulse/channel views.
        let results = SimulationResults {
            statistical_eye: Some(StatisticalEye::new(8)),
            eye_metrics: None,
            eye_diagram: None,
            channel_pulse: None,
            output_waveform: None,
            channel_response: None,
            training_result: None,
        };
        let config = synthetic_config();
        let app = App::new(&results, &config);
        assert_eq!(app.tabs.len(), 2); // Summary + statistical eye

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
    }

    #[test]
    fn test_decimate_minmax_preserves_peak() {
        let mut samples = vec![0.0; 10_000];
        samples[5_000] = 3.5; // lone spike must survive decimation
        let pts = decimate_minmax(&samples, 0.0, 1e-12, 100);
        assert!(pts.len() <= 201);
        let max = pts.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        assert!((max - 3.5).abs() < 1e-12, "peak lost in decimation: {}", max);
    }

    #[test]
    fn test_density_color_monotonic_ramp() {
        let ln_max = (1.0 + 1000.0f64).ln();
        // Low density is blue-ish, max density is red-ish.
        let Color::Rgb(r0, _, b0) = density_color(1, ln_max) else {
            panic!("expected RGB")
        };
        let Color::Rgb(r1, _, b1) = density_color(1000, ln_max) else {
            panic!("expected RGB")
        };
        assert!(b0 > r0, "low density should be blue-dominant");
        assert!(r1 > b1, "high density should be red-dominant");
    }
}
