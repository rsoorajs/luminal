use std::fmt::Debug;

use crate::prelude::*;
use candle_core::{Device, Tensor};
use rand::{Rng, rng};

#[test]
fn simple() {
    let mut cx = Graph::new();
    let b = cx.tensor(3);
    let c = cx.tensor(3);
    let g = cx.tensor(3);
    let e = cx.tensor(3);

    let a = (b * c + g).output();
    let d = (b * c / e).sin().output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(b.id, vec![1.0, 2.0, 3.0].into());
    rt.set_data(c.id, vec![1.0, 2.0, 3.0].into());
    rt.set_data(g.id, vec![1.0, 2.0, 3.0].into());
    rt.set_data(e.id, vec![1.0, 2.0, 3.0].into());

    rt.execute(&cx.dyn_map);

    // Reference
    let device = Device::Cpu;
    let ref_b = Tensor::new(vec![1_f32, 2_f32, 3_f32], &device).unwrap();
    let ref_c = Tensor::new(vec![1_f32, 2_f32, 3_f32], &device).unwrap();
    let ref_g = Tensor::new(vec![1_f32, 2_f32, 3_f32], &device).unwrap();
    let ref_e = Tensor::new(vec![1_f32, 2_f32, 3_f32], &device).unwrap();

    let ref_a = (ref_b.clone() * ref_c.clone() + ref_g).unwrap();
    let ref_d = (ref_b * ref_c / ref_e).unwrap().sin().unwrap();

    assert_eq!(*rt.get_f32(a.id), ref_a.to_vec1::<f32>().unwrap());
    assert_eq!(*rt.get_f32(d.id), ref_d.to_vec1::<f32>().unwrap());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let b = cx.tensor((3, 1));
    let c = cx.tensor((1, 4));

    let a = b.matmul(c).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(b.id, vec![1.0, 2.0, 3.0].into());
    rt.set_data(c.id, vec![1.0, 2.0, 3.0, 3.0].into());
    rt.execute(&cx.dyn_map);

    // Reference
    let device = Device::Cpu;
    let ref_b = Tensor::new(vec![vec![1_f32], vec![2_f32], vec![3_f32]], &device).unwrap();
    let ref_c = Tensor::new(vec![vec![1_f32, 2_f32, 3_f32, 3_f32]], &device).unwrap();
    let ref_a = ref_b.matmul(&ref_c).unwrap();
    assert_eq!(
        *rt.get_f32(a.id),
        ref_a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}

#[test]
fn test_shapes() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 2));
    let b = (a.permute((1, 0)) * 1.0).output();
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(a.id, vec![1.0, 2.0, 3.0, 4.0].into());
    rt.execute(&cx.dyn_map);

    assert_exact(rt.get_f32(b.id), &[1., 3., 2., 4.]);
}

#[test]
fn test_top_k_filter() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 6));
    let kth_largest = a.gather(a.topk_indexes(3, 1).slice((.., 2..3)).squeeze(1));
    let mask = a.ge(kth_largest.expand_dim(1, 6));
    let filtered = (a * mask).output();
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(
        a.id,
        vec![1.0, 2.0, 3.0, 4.0, 5., 6., 1.0, 2.0, 3.0, 4.0, 5., 6.].into(),
    );
    rt.execute(&cx.dyn_map);
    assert_eq!(
        *rt.get_f32(filtered.id),
        vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0]
    );
}

/// Ensure two arrays are nearly equal
pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

/// Ensure two arrays are nearly equal to a decimal place
pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

/// Ensure two arrays are exactly equal
pub fn assert_exact<T: PartialEq + Debug>(a_vec: &[T], b_vec: &[T]) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if a != b {
            panic!("{a:?} is not equal to {b:?}, index {i}");
        }
    }
}

pub fn random_array<const N: usize>() -> [f32; N] {
    let mut rng = rng();
    random_array_rng(&mut rng)
}

pub fn random_array_rng<const N: usize, R: Rng>(rng: &mut R) -> [f32; N] {
    let mut arr = [0.; N];
    for i in &mut arr {
        *i = rng.random_range(-0.5..0.5);
    }
    arr
}

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = rng();
    random_vec_rng(n, &mut rng)
}

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}
