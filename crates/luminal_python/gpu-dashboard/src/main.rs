use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use nvml_wrapper::Nvml;
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, Gauge, GraphType,
        Paragraph, Sparkline,
    },
    DefaultTerminal, Frame,
};

const HISTORY_LEN: usize = 120;
const POLL_INTERVAL: Duration = Duration::from_millis(500);

struct GpuSnapshot {
    name: String,
    mem_used_mib: u64,
    mem_total_mib: u64,
    mem_free_mib: u64,
    gpu_util: u32,
    mem_util: u32,
    temperature: u32,
    power_draw_mw: u32,
    power_limit_mw: u32,
    fan_speed: Option<u32>,
    sm_clock_mhz: u32,
    mem_clock_mhz: u32,
    pcie_tx_kbps: Option<u64>,
    pcie_rx_kbps: Option<u64>,
}

struct GpuState {
    current: GpuSnapshot,
    mem_history: VecDeque<u64>,
    gpu_util_history: VecDeque<u64>,
    temp_history: VecDeque<u64>,
    power_history: VecDeque<u64>,
    peak_mem_mib: u64,
}

struct App {
    gpus: Vec<GpuState>,
    selected_gpu: usize,
    nvml: Nvml,
    tick: u64,
}

impl App {
    fn new() -> Result<Self> {
        let nvml = Nvml::init()?;
        let count = nvml.device_count()?;
        let mut gpus = Vec::new();

        for i in 0..count {
            let device = nvml.device_by_index(i)?;
            let snapshot = Self::read_device(&device)?;
            gpus.push(GpuState {
                peak_mem_mib: snapshot.mem_used_mib,
                current: snapshot,
                mem_history: VecDeque::from(vec![0; HISTORY_LEN]),
                gpu_util_history: VecDeque::from(vec![0; HISTORY_LEN]),
                temp_history: VecDeque::from(vec![0; HISTORY_LEN]),
                power_history: VecDeque::from(vec![0; HISTORY_LEN]),
            });
        }

        Ok(Self {
            gpus,
            selected_gpu: 0,
            nvml,
            tick: 0,
        })
    }

    fn read_device(
        device: &nvml_wrapper::Device,
    ) -> Result<GpuSnapshot> {
        let name = device.name()?;
        let mem = device.memory_info()?;
        let util = device.utilization_rates()?;
        let temp = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)?;
        let power = device.power_usage()?; // milliwatts
        let power_limit = device.enforced_power_limit()?;
        let fan = device.fan_speed(0).ok();
        let sm_clock = device
            .clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM)
            .unwrap_or(0);
        let mem_clock = device
            .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
            .unwrap_or(0);

        let pcie_tx = device.pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send).ok().map(|v| v as u64);
        let pcie_rx = device.pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive).ok().map(|v| v as u64);

        Ok(GpuSnapshot {
            name,
            mem_used_mib: mem.used / (1024 * 1024),
            mem_total_mib: mem.total / (1024 * 1024),
            mem_free_mib: mem.free / (1024 * 1024),
            gpu_util: util.gpu,
            mem_util: util.memory,
            temperature: temp,
            power_draw_mw: power,
            power_limit_mw: power_limit,
            fan_speed: fan,
            sm_clock_mhz: sm_clock,
            mem_clock_mhz: mem_clock,
            pcie_tx_kbps: pcie_tx,
            pcie_rx_kbps: pcie_rx,
        })
    }

    fn poll(&mut self) -> Result<()> {
        self.tick += 1;
        let count = self.nvml.device_count()?;
        for i in 0..count {
            let device = self.nvml.device_by_index(i)?;
            let snap = Self::read_device(&device)?;
            let state = &mut self.gpus[i as usize];

            state.mem_history.pop_front();
            state.mem_history.push_back(snap.mem_used_mib);

            state.gpu_util_history.pop_front();
            state.gpu_util_history.push_back(snap.gpu_util as u64);

            state.temp_history.pop_front();
            state.temp_history.push_back(snap.temperature as u64);

            state.power_history.pop_front();
            state.power_history.push_back(snap.power_draw_mw as u64 / 1000);

            if snap.mem_used_mib > state.peak_mem_mib {
                state.peak_mem_mib = snap.mem_used_mib;
            }
            state.current = snap;
        }
        Ok(())
    }

    fn draw(&self, frame: &mut Frame) {
        let gpu = &self.gpus[self.selected_gpu];
        let snap = &gpu.current;

        // Main layout: title bar, then body
        let outer = Layout::vertical([Constraint::Length(3), Constraint::Min(0)])
            .split(frame.area());

        // Title bar
        let gpu_tabs: Vec<Span> = self
            .gpus
            .iter()
            .enumerate()
            .map(|(i, _g)| {
                let label = format!(" GPU {} ", i);
                if i == self.selected_gpu {
                    Span::styled(label, Style::default().fg(Color::Black).bg(Color::Cyan).bold())
                } else {
                    Span::styled(label, Style::default().fg(Color::DarkGray))
                }
            })
            .collect();

        let mut title_spans = vec![
            Span::styled(" GPU Dashboard ", Style::default().fg(Color::Cyan).bold()),
            Span::raw("| "),
        ];
        title_spans.extend(gpu_tabs);
        title_spans.push(Span::styled(
            "  [Tab] switch  [q] quit",
            Style::default().fg(Color::DarkGray),
        ));

        let title = Paragraph::new(Line::from(title_spans))
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
        frame.render_widget(title, outer[0]);

        // Body: left panel (info + gauges) | right panel (charts)
        let body = Layout::horizontal([Constraint::Length(42), Constraint::Min(0)])
            .split(outer[1]);

        // Left panel: device info, memory gauge, GPU util gauge, power gauge, details
        let left = Layout::vertical([
            Constraint::Length(5),  // device info
            Constraint::Length(4),  // memory gauge
            Constraint::Length(4),  // GPU util gauge
            Constraint::Length(4),  // power gauge
            Constraint::Length(8),  // memory breakdown bar chart
            Constraint::Min(0),    // details
        ])
        .split(body[0]);

        // Device info
        let info_text = vec![
            Line::from(vec![
                Span::styled("  Device: ", Style::default().fg(Color::DarkGray)),
                Span::styled(&snap.name, Style::default().fg(Color::White).bold()),
            ]),
            Line::from(vec![
                Span::styled("  Clocks: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("SM {} MHz / Mem {} MHz", snap.sm_clock_mhz, snap.mem_clock_mhz),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Temp:   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}C", snap.temperature),
                    temp_color(snap.temperature),
                ),
                Span::raw("  "),
                Span::styled("Fan: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    snap.fan_speed.map_or("N/A".into(), |f| format!("{}%", f)),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];
        let info = Paragraph::new(info_text)
            .block(Block::default().title(" Device Info ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
        frame.render_widget(info, left[0]);

        // Memory gauge
        let mem_pct = if snap.mem_total_mib > 0 {
            (snap.mem_used_mib as f64 / snap.mem_total_mib as f64).min(1.0)
        } else {
            0.0
        };
        let mem_label = format!(
            "{} / {} MiB ({:.1}%)",
            snap.mem_used_mib,
            snap.mem_total_mib,
            mem_pct * 100.0
        );
        let mem_gauge = Gauge::default()
            .block(Block::default().title(" Memory ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(mem_bar_color(mem_pct)).bg(Color::Black))
            .ratio(mem_pct)
            .label(mem_label);
        frame.render_widget(mem_gauge, left[1]);

        // GPU utilization gauge
        let gpu_pct = snap.gpu_util as f64 / 100.0;
        let gpu_label = format!("{}%", snap.gpu_util);
        let gpu_gauge = Gauge::default()
            .block(Block::default().title(" GPU Utilization ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
            .ratio(gpu_pct)
            .label(gpu_label);
        frame.render_widget(gpu_gauge, left[2]);

        // Power gauge
        let power_w = snap.power_draw_mw as f64 / 1000.0;
        let power_limit_w = snap.power_limit_mw as f64 / 1000.0;
        let power_pct = if power_limit_w > 0.0 {
            (power_w / power_limit_w).min(1.0)
        } else {
            0.0
        };
        let power_label = format!("{:.0} / {:.0} W", power_w, power_limit_w);
        let power_gauge = Gauge::default()
            .block(Block::default().title(" Power ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .gauge_style(Style::default().fg(Color::Magenta).bg(Color::Black))
            .ratio(power_pct)
            .label(power_label);
        frame.render_widget(power_gauge, left[3]);

        // Memory breakdown bar chart
        let mem_bars = BarGroup::default().bars(&[
            Bar::default()
                .value(snap.mem_used_mib)
                .label("Used".into())
                .style(Style::default().fg(Color::Red)),
            Bar::default()
                .value(snap.mem_free_mib)
                .label("Free".into())
                .style(Style::default().fg(Color::Green)),
        ]);
        let mem_barchart = BarChart::default()
            .block(Block::default().title(" Memory Breakdown (MiB) ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)))
            .data(mem_bars)
            .bar_width(12)
            .bar_gap(3)
            .value_style(Style::default().fg(Color::White).bold());
        frame.render_widget(mem_barchart, left[4]);

        // Details
        let mut detail_lines = vec![
            Line::from(vec![
                Span::styled("  Peak Mem:  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{} MiB", gpu.peak_mem_mib),
                    Style::default().fg(Color::Yellow).bold(),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Mem BW:    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}%", snap.mem_util),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];
        if let (Some(tx), Some(rx)) = (snap.pcie_tx_kbps, snap.pcie_rx_kbps) {
            detail_lines.push(Line::from(vec![
                Span::styled("  PCIe TX:   ", Style::default().fg(Color::DarkGray)),
                Span::styled(format_throughput(tx), Style::default().fg(Color::Cyan)),
            ]));
            detail_lines.push(Line::from(vec![
                Span::styled("  PCIe RX:   ", Style::default().fg(Color::DarkGray)),
                Span::styled(format_throughput(rx), Style::default().fg(Color::Cyan)),
            ]));
        }
        let details = Paragraph::new(detail_lines)
            .block(Block::default().title(" Details ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
        frame.render_widget(details, left[5]);

        // Right panel: sparklines / time-series charts
        let right = Layout::vertical([
            Constraint::Length(5),  // memory sparkline
            Constraint::Length(5),  // GPU util sparkline
            Constraint::Length(5),  // temperature sparkline
            Constraint::Length(5),  // power sparkline
            Constraint::Min(0),    // memory time-series chart
        ])
        .split(body[1]);

        // Memory sparkline
        let mem_data: Vec<u64> = gpu.mem_history.iter().copied().collect();
        let mem_spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(
                        " Memory Usage  [{} MiB / {} MiB] ",
                        snap.mem_used_mib, snap.mem_total_mib
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .data(&mem_data)
            .max(snap.mem_total_mib)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(mem_spark, right[0]);

        // GPU util sparkline
        let util_data: Vec<u64> = gpu.gpu_util_history.iter().copied().collect();
        let util_spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(" GPU Utilization  [{}%] ", snap.gpu_util))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .data(&util_data)
            .max(100)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(util_spark, right[1]);

        // Temperature sparkline
        let temp_data: Vec<u64> = gpu.temp_history.iter().copied().collect();
        let temp_spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(" Temperature  [{}C] ", snap.temperature))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .data(&temp_data)
            .max(100)
            .style(temp_color_simple(snap.temperature));
        frame.render_widget(temp_spark, right[2]);

        // Power sparkline
        let power_data: Vec<u64> = gpu.power_history.iter().copied().collect();
        let power_limit_w_u64 = snap.power_limit_mw as u64 / 1000;
        let power_spark = Sparkline::default()
            .block(
                Block::default()
                    .title(format!(
                        " Power  [{:.0}W / {:.0}W] ",
                        power_w, power_limit_w
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .data(&power_data)
            .max(power_limit_w_u64)
            .style(Style::default().fg(Color::Magenta));
        frame.render_widget(power_spark, right[3]);

        // Memory time-series chart (detailed line chart)
        let chart_data: Vec<(f64, f64)> = gpu
            .mem_history
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64, v as f64))
            .collect();

        let peak_data: Vec<(f64, f64)> = (0..HISTORY_LEN)
            .map(|i| (i as f64, gpu.peak_mem_mib as f64))
            .collect();

        let datasets = vec![
            Dataset::default()
                .name("Used")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&chart_data),
            Dataset::default()
                .name("Peak")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM))
                .data(&peak_data),
        ];

        let x_axis = Axis::default()
            .title("Time (last 60s)")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([0.0, HISTORY_LEN as f64])
            .labels(vec![Line::from("60s ago"), Line::from("30s"), Line::from("now")]);

        let y_axis = Axis::default()
            .title("MiB")
            .style(Style::default().fg(Color::DarkGray))
            .bounds([0.0, snap.mem_total_mib as f64])
            .labels(vec![
                Line::from("0"),
                Line::from(format!("{}", snap.mem_total_mib / 2)),
                Line::from(format!("{}", snap.mem_total_mib)),
            ]);

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title(" Memory History ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .x_axis(x_axis)
            .y_axis(y_axis);
        frame.render_widget(chart, right[4]);
    }
}

fn temp_color(temp: u32) -> Style {
    match temp {
        0..=50 => Style::default().fg(Color::Green),
        51..=75 => Style::default().fg(Color::Yellow),
        _ => Style::default().fg(Color::Red).bold(),
    }
}

fn temp_color_simple(temp: u32) -> Style {
    match temp {
        0..=50 => Style::default().fg(Color::Green),
        51..=75 => Style::default().fg(Color::Yellow),
        _ => Style::default().fg(Color::Red),
    }
}

fn mem_bar_color(pct: f64) -> Color {
    if pct < 0.5 {
        Color::Green
    } else if pct < 0.8 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn format_throughput(kbps: u64) -> String {
    if kbps > 1_000_000 {
        format!("{:.1} GB/s", kbps as f64 / 1_000_000.0)
    } else if kbps > 1_000 {
        format!("{:.1} MB/s", kbps as f64 / 1_000.0)
    } else {
        format!("{} KB/s", kbps)
    }
}

fn main() -> Result<()> {
    let mut app = App::new()?;

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    let result = run(&mut terminal, &mut app);

    ratatui::restore();
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    result
}

fn run(terminal: &mut DefaultTerminal, app: &mut App) -> Result<()> {
    let mut last_poll = Instant::now();

    loop {
        terminal.draw(|frame| app.draw(frame))?;

        let timeout = POLL_INTERVAL
            .checked_sub(last_poll.elapsed())
            .unwrap_or(Duration::ZERO);

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Tab => {
                            if !app.gpus.is_empty() {
                                app.selected_gpu =
                                    (app.selected_gpu + 1) % app.gpus.len();
                            }
                        }
                        KeyCode::BackTab => {
                            if !app.gpus.is_empty() {
                                app.selected_gpu = app
                                    .selected_gpu
                                    .checked_sub(1)
                                    .unwrap_or(app.gpus.len() - 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if last_poll.elapsed() >= POLL_INTERVAL {
            app.poll()?;
            last_poll = Instant::now();
        }
    }
}
