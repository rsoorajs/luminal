//! Benchmark result metrics

use serde::{Deserialize, Serialize};

/// Result of a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Backend name (e.g., "metal", "cuda")
    pub backend: String,
    /// Benchmark pattern name (e.g., "add_vec")
    pub benchmark: String,
    /// Size label (e.g., "1m")
    pub size_label: String,
    /// Actual size value
    pub size_value: usize,
    /// Mean execution time in microseconds
    pub mean_us: f64,
    /// Standard deviation in microseconds
    pub std_us: f64,
    /// Throughput in GB/s (if applicable)
    pub throughput_gbps: Option<f64>,
    /// Memory Bandwidth Utilization (if peak bandwidth known)
    pub mbu: Option<f64>,
}

impl BenchResult {
    /// Calculate throughput given bytes transferred
    pub fn with_throughput(mut self, bytes: usize) -> Self {
        if self.mean_us > 0.0 {
            // bytes / microseconds = MB/s, then convert to GB/s
            self.throughput_gbps = Some((bytes as f64) / self.mean_us / 1000.0);
        }
        self
    }

    /// Calculate MBU given peak bandwidth
    pub fn with_mbu(mut self, peak_bandwidth_gbps: f64) -> Self {
        if let Some(throughput) = self.throughput_gbps {
            self.mbu = Some(throughput / peak_bandwidth_gbps * 100.0);
        }
        self
    }
}

/// Collection of benchmark results for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub backend: String,
    pub hardware: String,
    pub results: Vec<BenchResult>,
}

impl BenchReport {
    pub fn new(backend: &str, hardware: &str) -> Self {
        Self {
            backend: backend.to_string(),
            hardware: hardware.to_string(),
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchResult) {
        self.results.push(result);
    }

    /// Export to JSON (for CI integration)
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}
