//! L1 Micro Benchmarks - Single operator performance tests

use crate::{BenchSize, BenchmarkPattern, MICRO_SIZES};
use luminal::prelude::*;

/// Vector addition benchmark: a + b
#[derive(Debug, Default)]
pub struct AddVec;

impl BenchmarkPattern for AddVec {
    fn name(&self) -> &'static str {
        "add_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = (a + b).output();
    }
}

/// Vector multiplication benchmark: a * b
#[derive(Debug, Default)]
pub struct MulVec;

impl BenchmarkPattern for MulVec {
    fn name(&self) -> &'static str {
        "mul_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = (a * b).output();
    }
}

/// Sum reduction benchmark: sum(a)
#[derive(Debug, Default)]
pub struct SumReduce;

impl BenchmarkPattern for SumReduce {
    fn name(&self) -> &'static str {
        "sum_reduce"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.sum(0).output();
    }
}

/// Max reduction benchmark: max(a)
#[derive(Debug, Default)]
pub struct MaxReduce;

impl BenchmarkPattern for MaxReduce {
    fn name(&self) -> &'static str {
        "max_reduce"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.max(0).output();
    }
}

/// Exp2 benchmark: 2^x
#[derive(Debug, Default)]
pub struct Exp2Bench;

impl BenchmarkPattern for Exp2Bench {
    fn name(&self) -> &'static str {
        "exp2"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.exp2().output();
    }
}

/// Get all micro benchmark patterns
pub fn all_micro_patterns() -> Vec<Box<dyn BenchmarkPattern>> {
    vec![
        Box::new(AddVec),
        Box::new(MulVec),
        Box::new(SumReduce),
        Box::new(MaxReduce),
        Box::new(Exp2Bench),
    ]
}

/// Calculate bytes transferred for a benchmark pattern
pub fn bytes_for_pattern(pattern_name: &str, size: usize) -> usize {
    let elem_size = std::mem::size_of::<f32>();
    match pattern_name {
        "add_vec" | "mul_vec" => {
            // Read 2 inputs + write 1 output = 3 * size * 4 bytes
            3 * size * elem_size
        }
        "sum_reduce" | "max_reduce" => {
            // Read 1 input + write 1 output (scalar)
            size * elem_size + elem_size
        }
        "exp2" => {
            // Read 1 input + write 1 output = 2 * size * 4 bytes
            2 * size * elem_size
        }
        _ => 0,
    }
}
