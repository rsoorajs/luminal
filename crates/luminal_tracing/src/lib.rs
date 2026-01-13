use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_perfetto_sdk_layer::NativeLayer;
use tracing_perfetto_sdk_schema::{
    DataSourceConfig, TraceConfig,
    trace_config::{BufferConfig, DataSource},
};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub struct TraceOptions {
    perfetto_file: Option<PathBuf>,
    env_filter: EnvFilter,
}

/// This is a convenience tracing subscriber with some opinionated defaults.
/// By default, it will try to use the RUST_LOG environment variable, falling back to "luminal=trace" if not set.
pub fn subscriber() -> TraceOptions {
    TraceOptions {
        perfetto_file: None,
        env_filter: EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| {
                EnvFilter::builder()
                    .parse("luminal=trace")
                    .expect("Invalid default filter")
            }),
    }
}

impl TraceOptions {
    pub fn perfetto(mut self, path: impl AsRef<Path>) -> Self {
        self.perfetto_file = Some(path.as_ref().to_path_buf());
        self
    }

    /// Override with a specific filter string (ignores RUST_LOG)
    pub fn env_filter(mut self, env_filter: impl ToString) -> Self {
        self.env_filter = EnvFilter::builder()
            .parse(env_filter.to_string())
            .expect("Invalid tracing env filter");
        self
    }

    /// Use RUST_LOG with a custom fallback if not set
    pub fn env_filter_or(mut self, fallback: impl ToString) -> Self {
        self.env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| {
                EnvFilter::builder()
                    .parse(fallback.to_string())
                    .expect("Invalid fallback filter")
            });
        self
    }

    /// Always use RUST_LOG (defaults to ERROR level if not set)
    pub fn from_env(mut self) -> Self {
        self.env_filter = EnvFilter::from_default_env();
        self
    }

    /// Install as the global tracing subscriber
    pub fn init(self) -> TraceSession {
        let filter = self.env_filter;

        if let Some(file_path) = self.perfetto_file {
            let file = File::create(&file_path).expect("Failed to create trace file");
            let (writer, guard) = tracing_appender::non_blocking(file);
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
                _guard: Some(guard),
                perfetto_path: Some(file_path),
            }
        } else {
            tracing_subscriber::registry()
                .with(filter.clone())
                .with(tracing_subscriber::fmt::layer())
                .init();
            TraceSession {
                perfetto_layer: None,
                _guard: None,
                perfetto_path: None,
            }
        }
    }
}

pub struct TraceSession {
    perfetto_layer: Option<NativeLayer<NonBlocking>>,
    _guard: Option<WorkerGuard>,
    pub perfetto_path: Option<PathBuf>,
}

impl TraceSession {
    pub fn flush(&self) {
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.flush(Duration::from_secs(5), Duration::from_secs(5));
        }
    }

    pub fn stop(&self) {
        self.flush();
        if let Some(layer) = &self.perfetto_layer {
            let _ = layer.stop();
        }
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
