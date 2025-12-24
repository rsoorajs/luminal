use itertools::Itertools;

use crate::{op, prelude::*};
use std::ops::{Add, Mul, Neg};

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Log2::default())
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Exp2::default())
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Recip::default())
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Sin::default())
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Sqrt::default())
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    pub fn graph_break(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::GraphBreak)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes)
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand(self.shape)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self.mean(axes).expand(self.shape)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        self.mean_norm(axes.to_axes()).std_norm(axes, epsilon)
    }

    /// Normalize the tensor along `axes` using an Lp norm.
    pub fn normalize(self, p: f32, axes: impl ToAxes, epsilon: f32) -> GraphTensor {
        let norm = self.abs().pow(p).sum(axes).pow(1.0 / p);
        self / norm.maximum_f32(epsilon).expand(self.shape)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand(self.shape);
        let exp = m.exp();
        exp / exp.sum(axes).expand(self.shape)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand(self.shape);
        m - m.exp().sum(axes.to_axes()).log().expand(m.shape)
    }

    /// Get the indicies of the max elements along an axis
    pub fn argmax(self, axis: usize) -> GraphTensor {
        // Get one-hot along last dimension
        let x_equal = self.eq(self.max(axis).expand(self.shape));
        // Create index arange for last dimension
        let r = self.graph().arange(self.dims()[axis]).cast(self.dtype);
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand(self.shape)).max(axis).cast(DType::Int)
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor {
        self.relu() + (-self).relu()
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor {
        self / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor {
        self.maximum_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        (1. + (-self).exp()).reciprocal()
    }

    /// The swish (aka silu) activation function
    pub fn swish(self) -> GraphTensor {
        self * self.sigmoid()
    }

    /// The silu (aka swish) activation function
    pub fn silu(self) -> GraphTensor {
        self.swish()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor {
        self.relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        0.5 * self * (1. + (0.7978845608 * self * (1. + 0.044715 * self * self)).tanh())
    }

    /// Compute the sorted indexes of this tensor along a certian axis
    pub fn argsort_indexes(self, axis: usize, ascending: bool) -> GraphTensor {
        // Compare all elements with all other elements by making a axis
        let ax_size = self.dims()[axis];
        let a = self.expand_dim(axis + 1, ax_size);
        let b = self.expand_dim(axis, ax_size) + 1e-9; // eps for stable sort
        let mut ind = a.lt(b).sum(axis).cast(DType::Int);
        if ascending {
            ind = -ind + (ax_size - 1);
        }
        ind.inverse_permutation(axis)
    }

    /// Sort the tensor along a certian axis
    pub fn argsort(self, axis: usize, ascending: bool) -> GraphTensor {
        self.gather(self.argsort_indexes(axis, ascending))
    }

    /// Sort and retrieve top-k **indexes**
    pub fn topk_indexes(self, k: usize, axis: usize) -> GraphTensor {
        self.argsort_indexes(axis, true).slice_along(..k, axis)
    }

    /// Apply a cumulative reduction operation along dimensions
    ///
    /// See `cumsum` or `cummax` for usage examples.
    pub fn cumop(
        mut self,
        axes: impl ToAxes,
        op: impl Fn(GraphTensor, &[usize]) -> GraphTensor,
    ) -> Self {
        let n_dims = self.shape.len();
        let axes = axes.to_axes();
        // Pad out length
        let mut kernel = vec![1.into(); n_dims];
        let mut padding = vec![(Expression::from(0), Expression::from(0)); n_dims];
        for &ax in &axes {
            let orig_length = self.dims()[ax];
            padding[ax] = (orig_length - 1, 0.into());
            kernel[ax] = orig_length;
        }
        self = self.pad(padding);
        // Unfold
        self = self.unfold(kernel, vec![1; n_dims], vec![1; n_dims]);
        // Remove non-cumulative dimensions
        for i in (0..n_dims).rev() {
            if !axes.contains(&i) {
                self = self.squeeze(n_dims + i);
            }
        }
        // apply operation along cumulative dimensions
        op(self, &(0..axes.len()).map(|ax| ax + n_dims).collect_vec())
    }

    /// Apply a cumulative sum along dimensions
    pub fn cumsum(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.sum(axes))
    }

    /// Apply a cumulative max along dimensions
    pub fn cummax(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.max(axes))
    }

    /// Apply a cumulative product along dimensions
    pub fn cumprod(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.prod(axes))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::BinaryHeap;

    use crate::{
        prelude::*,
        tests::{assert_close, random_vec},
    };
    use candle_core::{Device, Tensor};
    use candle_nn::ops::softmax;
    use itertools::Itertools;
    use ordered_float::NotNan;

    pub fn test_unary(
        shape: impl ToShape,
        func: fn(GraphTensor) -> GraphTensor,
        ref_func: fn(Tensor) -> Tensor,
    ) {
        let shape = shape
            .to_shape()
            .into_iter()
            .map(|e| e.to_usize().unwrap())
            .collect_vec();
        let mut cx = Graph::new();
        let a = cx.tensor(shape.clone());
        let b = func(a).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        let v = random_vec(shape.iter().copied().product());
        rt.set_data(a.id, v.clone().into());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::new(v, &device).unwrap().reshape(shape).unwrap();
        let ref_b = ref_func(ref_a).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
    }

    #[test]
    fn test_exp() {
        test_unary(27, |a| a.exp(), |a| a.exp().unwrap());
    }
    #[test]
    fn test_log() {
        test_unary(27, |a| a.log(), |a| a.log().unwrap());
    }
    #[test]
    fn test_sin() {
        test_unary(27, |a| a.sin(), |a| a.sin().unwrap());
    }
    #[test]
    fn test_cos() {
        test_unary(27, |a| a.cos(), |a| a.cos().unwrap());
    }
    #[test]
    fn test_activations() {
        test_unary(27, |a| a.relu(), |a| a.relu().unwrap());
        test_unary(27, |a| a.gelu(), |a| a.gelu().unwrap());
        test_unary(27, |a| a.swish(), |a| a.silu().unwrap());
        test_unary(27, |a| a.tanh(), |a| a.tanh().unwrap());
    }
    #[test]
    fn test_recip() {
        test_unary(27, |a| a.reciprocal(), |a| a.recip().unwrap());
    }
    #[test]
    fn test_sqrt() {
        test_unary(27, |a| a.sqrt(), |a| a.sqrt().unwrap());
    }
    #[test]
    fn test_square() {
        test_unary(27, |a| a.square(), |a| a.powf(2.0).unwrap());
    }
    #[test]
    fn test_softmax() {
        test_unary(27, |a| a.softmax(0), |a| softmax(&a, 0).unwrap());
    }

    #[test]
    fn test_layer_norm() {
        test_unary(
            27,
            |a| a.layer_norm(0, 1e-5),
            |a| {
                let meaned = (a.clone() - a.mean(0).unwrap().broadcast_as(27)).unwrap();
                meaned
                    .powf(2.0)
                    .unwrap()
                    .mean(0)
                    .unwrap()
                    .add(&Tensor::new(1e-5_f32, a.device()).unwrap())
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap()
                    .broadcast_as(27)
                    .unwrap()
                    .mul(&meaned)
                    .unwrap()
            },
        );
    }
    #[test]
    fn test_cumsum() {
        test_unary(27, |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
        test_unary((27, 63), |a| a.cumsum(1), |a| a.cumsum(1).unwrap());
        test_unary((27, 63), |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
    }
    #[test]
    fn test_argmax() {
        test_unary(
            (9, 27),
            |a| a.argmax(1).cast(DType::F32),
            |a| {
                a.argmax(1)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            },
        );
        test_unary(
            (9, 27),
            |a| a.argmax(0).cast(DType::F32),
            |a| {
                a.argmax(0)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            },
        );
    }
    #[test]
    fn test_topk() {
        pub fn topk_sorted_indices(x: &[f32], k: usize) -> Vec<usize> {
            if k == 0 {
                return Vec::new();
            }

            let mut heap: BinaryHeap<std::cmp::Reverse<(NotNan<f32>, usize)>> =
                BinaryHeap::with_capacity(k);

            for (i, &v) in x.iter().enumerate() {
                let v = NotNan::new(v).expect("NaN encountered in topk");
                if heap.len() < k {
                    heap.push(std::cmp::Reverse((v, i)));
                } else if let Some(&std::cmp::Reverse((min_v, _))) = heap.peek() {
                    if v > min_v {
                        heap.pop();
                        heap.push(std::cmp::Reverse((v, i)));
                    }
                }
            }

            let mut out: Vec<(NotNan<f32>, usize)> =
                heap.into_iter().map(|std::cmp::Reverse(t)| t).collect();

            out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            out.into_iter().map(|(_, i)| i).collect()
        }
        test_unary(
            (10, 9),
            |a| a.topk_indexes(5, 1).cast(DType::F32) * 1.0,
            |a| {
                let data = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let topk = data
                    .chunks_exact(9)
                    .flat_map(|c| topk_sorted_indices(c, 5))
                    .map(|i| i as f32)
                    .collect_vec();
                Tensor::new(topk, a.device()).unwrap()
            },
        );
    }
}
