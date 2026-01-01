use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use tracing_appender::non_blocking::{self, WorkerGuard};
use tracing_perfetto_sdk_layer::NativeLayer;
use tracing_perfetto_sdk_schema::{
    DataSourceConfig, TraceConfig,
    trace_config::{BufferConfig, DataSource},
};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub enum TraceSink {
    PerfettoFile { path: PathBuf },
    Stdout,
    Disabled,
}

pub struct TraceOptions {
    pub sink: TraceSink,
    pub env_filter: String,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self {
            sink: TraceSink::Stdout,
            env_filter: "luminal=trace".to_string(),
        }
    }
}

pub struct TraceSession {
    perfetto_layer: Option<tracing_perfetto_sdk_layer::LayerHandle>,
    guard: Option<WorkerGuard>,
    perfetto_path: Option<PathBuf>,
}

impl TraceSession {
    pub fn flush(&self) {
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.flush(Duration::from_secs(5), Duration::from_secs(5));
        }
    }

    pub fn stop(&self) {
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.stop();
        }
    }

    pub fn perfetto_path(&self) -> Option<&Path> {
        self.perfetto_path.as_deref()
    }
}

pub fn init(options: TraceOptions) -> TraceSession {
    let filter = EnvFilter::builder()
        .parse(options.env_filter)
        .expect("Invalid tracing env filter");

    match options.sink {
        TraceSink::PerfettoFile { path } => init_perfetto_file(&filter, path),
        TraceSink::Stdout => init_stdout(&filter),
        TraceSink::Disabled => {
            tracing_subscriber::registry().with(filter).init();
            TraceSession {
                perfetto_layer: None,
                guard: None,
                perfetto_path: None,
            }
        }
    }
}

fn init_perfetto_file(filter: &EnvFilter, path: PathBuf) -> TraceSession {
    let file = File::create(&path).expect("Failed to create trace file");
    let (writer, guard) = non_blocking(file);
    let layer = NativeLayer::from_config(default_perfetto_config(), writer)
        .build()
        .expect("Failed to build perfetto layer");
    let handle = layer.clone();
    tracing_subscriber::registry()
        .with(filter.clone())
        .with(layer)
        .init();
    TraceSession {
        perfetto_layer: Some(handle),
        guard: Some(guard),
        perfetto_path: Some(path),
    }
}

fn init_stdout(filter: &EnvFilter) -> TraceSession {
    tracing_subscriber::registry()
        .with(filter.clone())
        .with(tracing_subscriber::fmt::layer())
        .init();
    TraceSession {
        perfetto_layer: None,
        guard: None,
        perfetto_path: None,
    }
}

fn default_perfetto_config() -> TraceConfig {
    TraceConfig {
        buffers: vec![BufferConfig {
            size_kb: Some(4096),
            ..Default::default()
        }],
        data_sources: vec![DataSource {
            config: Some(DataSourceConfig {
                name: Some("rust_tracing".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

pub fn trace_file_path(path: impl AsRef<Path>) -> TraceSink {
    TraceSink::PerfettoFile {
        path: path.as_ref().to_path_buf(),
    }
}
