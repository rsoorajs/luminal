//! Benchmark metrics and mapping
//!
//! Provides a mapping from benchmark names to their constant metrics (bytes, flops).
//! Combined with Criterion's time measurements, this allows computing derived metrics
//! like throughput, MBU, and MFU.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Constant metrics for a single benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetrics {
    /// Total bytes transferred (loaded + stored)
    pub bytes: usize,
    /// Bytes loaded from memory
    pub bytes_loaded: usize,
    /// Bytes stored to memory
    pub bytes_stored: usize,
    /// Floating-point operations
    pub flops: usize,
}

impl BenchMetrics {
    pub fn new(bytes_loaded: usize, bytes_stored: usize, flops: usize) -> Self {
        Self {
            bytes: bytes_loaded + bytes_stored,
            bytes_loaded,
            bytes_stored,
            flops,
        }
    }

    /// Calculate throughput in GB/s given execution time in microseconds
    pub fn throughput_gbps(&self, time_us: f64) -> f64 {
        if time_us <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / time_us / 1000.0
    }

    /// Calculate TFLOPS given execution time in microseconds
    pub fn tflops(&self, time_us: f64) -> f64 {
        if time_us <= 0.0 {
            return 0.0;
        }
        self.flops as f64 / time_us / 1_000_000.0
    }

    /// Calculate MBU given execution time and peak bandwidth
    pub fn mbu(&self, time_us: f64, peak_bandwidth_gbps: f64) -> f64 {
        self.throughput_gbps(time_us) / peak_bandwidth_gbps * 100.0
    }

    /// Calculate MFU given execution time and peak TFLOPS
    pub fn mfu(&self, time_us: f64, peak_tflops: f64) -> f64 {
        self.tflops(time_us) / peak_tflops * 100.0
    }
}

/// Hardware specifications for a benchmark target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub device_name: String,
    pub memory_gb: f64,
    pub peak_bandwidth_gbps: f64,
    pub peak_tflops: f64,
}

/// Complete benchmark metrics mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetricsMap {
    /// Hardware specifications
    pub hardware: HardwareSpec,
    /// Mapping from "pattern/size" to metrics
    pub benchmarks: HashMap<String, BenchMetrics>,
}

impl BenchMetricsMap {
    pub fn new(hardware: HardwareSpec) -> Self {
        Self {
            hardware,
            benchmarks: HashMap::new(),
        }
    }

    /// Add metrics for a benchmark
    pub fn add(&mut self, pattern: &str, size: &str, metrics: BenchMetrics) {
        let key = format!("{}/{}", pattern, size);
        self.benchmarks.insert(key, metrics);
    }

    /// Get metrics for a benchmark
    pub fn get(&self, pattern: &str, size: &str) -> Option<&BenchMetrics> {
        let key = format!("{}/{}", pattern, size);
        self.benchmarks.get(&key)
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json();
        std::fs::write(path, json)
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ============================================================================
// Legacy types (kept for compatibility)
// ============================================================================

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
