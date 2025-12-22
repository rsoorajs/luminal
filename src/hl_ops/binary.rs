use crate::op;
use crate::prelude::*;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::RemAssign;
use std::ops::SubAssign;
use std::ops::{Add, Div, Mul, Rem, Sub};

impl Add for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        // assert_eq!(
        //     self.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     rhs.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     "Dims must match to add tensors."
        // );
        let new_id = self
            .graph()
            .add_op(op::Add::default())
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl Add<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        rhs + self
    }
}

impl<T> AddAssign<T> for GraphTensor
where
    GraphTensor: Add<T, Output = GraphTensor>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl Sub for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl Sub<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl<T> SubAssign<T> for GraphTensor
where
    GraphTensor: Sub<T, Output = GraphTensor>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        // assert_eq!(
        //     self.dims(),
        //     rhs.dims(),
        //     "Dims must match to multiply tensors."
        // );
        let new_id = self
            .graph()
            .add_op(op::Mul::default())
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl Mul<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        rhs * self
    }
}

impl<T> MulAssign<T> for GraphTensor
where
    GraphTensor: Mul<T, Output = GraphTensor>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.reciprocal()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.reciprocal()
    }
}

impl<T> DivAssign<T> for GraphTensor
where
    GraphTensor: Div<T, Output = GraphTensor>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl Rem<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: GraphTensor) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to mod tensors.");
        let new_id = self
            .graph()
            .add_op(op::Mod::default())
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }
}

impl<T> RemAssign<T> for GraphTensor
where
    GraphTensor: Rem<T, Output = GraphTensor>,
{
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

impl Add<f32> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: f32) -> Self::Output {
        self + self.graph().constant_float(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self.graph().constant(rhs).expand(self.shape)
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant_float(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self.graph().constant(rhs).expand(self.shape)
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self.graph().constant_float(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self.graph().constant(rhs).expand(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        self * self.graph().constant_float(rhs.recip()).expand(self.shape)
    }
}

impl<S: Into<Expression>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand(self.shape)
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant_float(rhs).expand(self.shape)
    }
}

impl<S: Into<Expression>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self.graph().constant(rhs).expand(self.shape)
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    pub fn lt(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to lt tensors.");
        let new_id = self
            .graph()
            .add_op(op::LessThan::default())
            .input(self.id, 0, self.shape)
            .input(rhs.id, 0, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        rhs.lt(self)
    }

    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        -self.gt(rhs) + 1.0
    }

    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        -self.lt(rhs) + 1.0
    }

    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        self.lt(rhs) + self.gt(rhs)
    }

    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        -self.ne(rhs) + 1.0
    }

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().log().mul(e).exp()
    }
}

// Clipping ops (minimum, maximum, clip)
impl GraphTensor {
    /// Take the elementwise maximum of two tensors
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs) * rhs) + (rhs.le(self) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        self.maximum(self.graph().constant_float(rhs).expand(self.shape))
    }

    /// Take the elementwise minimum of two tensors
    pub fn minimum(self, rhs: GraphTensor) -> GraphTensor {
        -(-self).maximum(-rhs)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn minimum_f32(self, rhs: f32) -> GraphTensor {
        -(-self).maximum_f32(-rhs)
    }

    /// Clip (clamp) a tensor into the range [`min`, `max`]
    pub fn clip(self, min: f32, max: f32) -> GraphTensor {
        self.maximum_f32(min).minimum_f32(max)
    }
}

pub trait F32Pow {
    fn pow(self, e: GraphTensor) -> GraphTensor;
}

impl F32Pow for f32 {
    fn pow(self, e: GraphTensor) -> GraphTensor {
        e.mul(self.abs().ln()).exp()
    }
}

// #[cfg(test)]
#[cfg(test)]
mod tests {
    use crate::{
        prelude::*,
        tests::{assert_close, random_vec},
    };
    use candle_core::{DType, Device, Tensor};

    const N_ELEMENTS: usize = 27;

    fn identity(v: Vec<f32>) -> Vec<f32> {
        v
    }

    fn shift_from_zero(v: Vec<f32>) -> Vec<f32> {
        v.into_iter()
            .map(|x| if x >= 0.0 { x + 1.0 } else { x - 1.0 })
            .collect()
    }

    fn test_binary<F, R>(func: F, ref_func: R)
    where
        F: Fn(GraphTensor, GraphTensor) -> GraphTensor,
        R: Fn(Tensor, Tensor, &Device) -> Tensor,
    {
        test_binary_with(func, ref_func, identity, identity);
    }

    fn test_binary_with<F, R>(
        func: F,
        ref_func: R,
        lhs_transform: fn(Vec<f32>) -> Vec<f32>,
        rhs_transform: fn(Vec<f32>) -> Vec<f32>,
    ) where
        F: Fn(GraphTensor, GraphTensor) -> GraphTensor,
        R: Fn(Tensor, Tensor, &Device) -> Tensor,
    {
        let mut cx = Graph::new();
        let a = cx.tensor(N_ELEMENTS);
        let b = cx.tensor(N_ELEMENTS);
        let c = func(a, b).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        let lhs_values = lhs_transform(random_vec(N_ELEMENTS));
        let rhs_values = rhs_transform(random_vec(N_ELEMENTS));
        rt.set_data(a.id, lhs_values.clone().into());
        rt.set_data(b.id, rhs_values.clone().into());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::new(lhs_values, &device).unwrap();
        let ref_b = Tensor::new(rhs_values, &device).unwrap();
        let ref_c = ref_func(ref_a, ref_b, &device);

        assert_close(rt.get_f32(c.id), &ref_c.to_vec1::<f32>().unwrap())
    }

    #[test]
    fn test_add() {
        test_binary(|a, b| a + b, |a, b, _| (&a + &b).unwrap());
    }

    #[test]
    fn test_sub() {
        test_binary(|a, b| a - b, |a, b, _| (&a - &b).unwrap());
    }

    #[test]
    fn test_mul() {
        test_binary(|a, b| a * b, |a, b, _| (&a * &b).unwrap());
    }

    #[test]
    fn test_div() {
        test_binary_with(
            |a, b| a / b,
            |a, b, _| (&a / &b).unwrap(),
            identity,
            shift_from_zero,
        );
    }

    #[test]
    fn test_maximum() {
        test_binary(|a, b| a.maximum(b), |a, b, _| a.maximum(&b).unwrap());
    }

    #[test]
    fn test_minimum() {
        test_binary(|a, b| a.minimum(b), |a, b, _| a.minimum(&b).unwrap());
    }

    #[test]
    fn test_mod() {
        test_binary_with(
            |a, b| a % b,
            |a, b, _| {
                let lhs = a.to_vec1::<f32>().unwrap();
                let rhs = b.to_vec1::<f32>().unwrap();
                let remainder: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x % y).collect();
                Tensor::from_vec(remainder, N_ELEMENTS, &Device::Cpu).unwrap()
            },
            identity,
            shift_from_zero,
        );
    }

    #[test]
    fn test_lt() {
        test_binary(
            |a, b| a.lt(b),
            |a, b, _| a.lt(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_gt() {
        test_binary(
            |a, b| a.gt(b),
            |a, b, _| a.gt(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_le() {
        test_binary(
            |a, b| a.le(b),
            |a, b, _| a.le(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_ge() {
        test_binary(
            |a, b| a.ge(b),
            |a, b, _| a.ge(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_ne() {
        test_binary(
            |a, b| a.ne(b),
            |a, b, _| a.ne(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_eq() {
        test_binary(
            |a, b| a.eq(b),
            |a, b, _| a.eq(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_pow() {
        test_binary_with(
            |a, _| a.pow(2.5f32),
            |a, _, _| a.powf(2.5f64).unwrap(),
            shift_from_zero,
            identity,
        );
    }

    #[test]
    fn test_clip() {
        test_binary_with(
            |a, _| a.clip(-0.25, 0.25),
            |a, _, _| a.clamp(-0.25, 0.25).unwrap(),
            identity,
            identity,
        );
    }

    #[test]
    fn test_maximum_f32() {
        test_binary_with(
            |a, _| a.maximum_f32(0.1),
            |a, _, _| {
                a.maximum(&Tensor::new(vec![0.1f32; N_ELEMENTS], &Device::Cpu).unwrap())
                    .unwrap()
            },
            identity,
            identity,
        );
    }

    #[test]
    fn test_minimum_f32() {
        test_binary_with(
            |a, _| a.minimum_f32(-0.1),
            |a, _, _| {
                a.minimum(&Tensor::new(vec![-0.1f32; N_ELEMENTS], &Device::Cpu).unwrap())
                    .unwrap()
            },
            identity,
            identity,
        );
    }
}
