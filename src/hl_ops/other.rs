use crate::{op::Constant, prelude::*};

// impl GraphTensor {
//     /// Cumulative sum last dimension
//     pub fn cumsum_last_dim(mut self) -> Self {
//         let axis = self.shape.len() - 1;
//         if !self.shape.is_contiguous() {
//             self = self.contiguous();
//         }
//         // Pad out length
//         let orig_length = self.dims()[axis];
//         self = self.pad_along(orig_length - 1, 0, axis).contiguous();

//         // Pool
//         self = self.pool_last_dim(orig_length, 1, 1);

//         // Sum Reduce along new dimension
//         self.sum(axis + 1)
//     }

//     /// Cumulative max last dimension
//     pub fn cummax_last_dim(mut self) -> Self {
//         let axis = self.shape.len() - 1;
//         if !self.shape.is_contiguous() {
//             self = self.contiguous();
//         }
//         // Pad out length
//         let orig_length = self.dims()[axis];
//         self.shape.padding[self.shape.indexes[axis]].0 = orig_length - 1;
//         self = self.contiguous();

//         // Pool
//         self = self.pool_last_dim(orig_length, 1, 1);
//         // Max Reduce along new dimension
//         self.max(axis + 1)
//     }

//     /// Cumulative product last dimension
//     pub fn cumprod_last_dim(self) -> Self {
//         self.log().cumsum_last_dim().exp()
//     }
// }

impl Graph {
    /// A scalar expression constant
    pub fn constant(&mut self, i: impl Into<Expression>) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Iota(i.into(), 1.into())).finish(),
            ShapeTracker::new(()),
            self,
            DType::Int,
        )
    }

    /// A scalar float constant
    pub fn constant_float(&mut self, i: f32) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Constant(i)).finish(),
            ShapeTracker::new(()),
            self,
            DType::F32,
        )
    }

    /// Iota expression
    pub fn iota(&mut self, i: impl Into<Expression>, shape: impl ToShape) -> GraphTensor {
        let sh = shape.to_shape();
        GraphTensor::from_id(
            self.add_op(Iota(i.into(), sh.iter().copied().product()))
                .finish(),
            ShapeTracker::new(sh),
            self,
            DType::Int,
        )
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<Expression>) -> GraphTensor {
        self.iota('z', to)
    }

    /// ARange from beginning to end
    pub fn arange_options(
        &mut self,
        start: impl Into<Expression>,
        end: impl Into<Expression>,
        step: impl Into<Expression>,
    ) -> GraphTensor {
        let (start, end, step) = (start.into(), end.into(), step.into());
        self.iota((Expression::from('z') * step) + start, (end - start) / step)
    }

    // /// Lower left-hand triangle of 1s. Currently required to be square
    // ///
    // /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    // pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
    //     let size = size.into();
    //     let horizontal = self.arange(size).expand_dim(0, size);
    //     let vertical = self.arange(size).expand_dim(1, size);

    //     (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    // }

    // /// Upper right-hand triangle of 1s
    // ///
    // /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    // pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
    //     let size = size.into();
    //     let horizontal = self.arange(size).expand_dim(0, size).contiguous();
    //     let vertical = self.arange(size).expand_dim(1, size).contiguous();

    //     (horizontal - (diagonal as f32 - 1.)).gt(vertical)
    // }
}

impl GraphTensor {
    pub fn cast(self, dtype: DType) -> GraphTensor {
        let id = self
            .graph()
            .add_op(Cast(dtype))
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref, dtype)
    }

    /// Sets this tensor's dtype without doing a cast
    pub fn as_dtype(mut self, dtype: DType) -> GraphTensor {
        self.dtype = dtype;
        if let Some(gmem) = self.graph().try_get_op_mut::<Input>(self.id) {
            gmem.dtype = dtype;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    // crate::test_imports!();
    // #[test]
    // fn test_arange() {
    //     let mut cx = Graph::new();

    //     let arange = cx.arange(10).retrieve();
    //     cx.execute();

    //     assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    // }

    // #[test]
    // fn test_arange_from_zero() {
    //     let mut cx = Graph::new();

    //     let tensor = cx.arange(5).retrieve();
    //     cx.execute();

    //     assert_eq!(tensor.data(), vec![0., 1., 2., 3., 4.]);
    // }

    // #[test]
    // fn test_arange_in_range() {
    //     let mut cx = Graph::new();

    //     let tensor = cx.arange_in_range(3, 8).retrieve();
    //     cx.execute();

    //     assert_eq!(tensor.data(), vec![3., 4., 5., 6., 7.]);
    // }

    // #[test]
    // fn test_arange_step_simple() {
    //     let mut cx = Graph::new();

    //     let tensor = cx.arange_step(1.0, 5.0, 1.0).retrieve();
    //     cx.execute();

    //     assert_eq!(tensor.data(), vec![1.0, 2.0, 3.0, 4.0]);
    // }

    // #[test]
    // fn test_arange_step_fractional() {
    //     let mut cx = Graph::new();

    //     let tensor = cx.arange_step(0.0, 1.0, 0.3).retrieve();
    //     cx.execute();

    //     // Should produce [0.0, 0.3, 0.6, 0.9] — note that 1.2 would be >= 1.0 so we stop before that.
    //     let expected = &[0.0, 0.3, 0.6, 0.9];

    //     // Floating point comparison with tolerance:
    //     assert_eq!(tensor.data().len(), expected.len());
    //     for (v, e) in tensor.data().iter().zip(expected.iter()) {
    //         assert!((v - e).abs() < 1e-5, "Expected {e}, got {v}");
    //     }
    // }

    // #[test]
    // #[should_panic(expected = "step must be positive")]
    // fn test_arange_step_zero_step_panics() {
    //     let mut cx = Graph::new();

    //     // Should panic because step is zero
    //     cx.arange_step(0.0, 5.0, 0.0);
    // }

    // #[test]
    // fn test_cumprod() {
    //     let mut cx = Graph::new();

    //     let a = cx.tensor(3).set(vec![3., 2., 5.]);
    //     let b = a.cumprod_last_dim().retrieve();
    //     cx.execute();

    //     assert_close(&b.data(), &[3., 6., 30.]);
    // }

    // #[test]
    // fn test_gather() {
    //     let mut cx = Graph::new();

    //     let matrix = cx.tensor((3, 2)).set(vec![1., 2., 3., 4., 5., 6.]);
    //     let indexes = cx.tensor(2).set(vec![2., 0.]);
    //     let result = matrix.gather(indexes).retrieve();

    //     cx.execute();

    //     assert_exact(&result.data(), &[5., 6., 1., 2.]);
    // }

    // #[test]
    // fn test_dyn_arange() {
    //     let mut cx = Graph::new();

    //     let arange = cx.arange('a').retrieve();
    //     cx.set_dyn_dim('a', 6);

    //     cx.execute();

    //     assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5.]);
    // }

    // #[test]
    // fn test_tril() {
    //     let mut cx = Graph::new();

    //     let triangle = cx.tril(5, 1).retrieve();

    //     cx.execute();

    //     assert_exact(
    //         &triangle.data(),
    //         &[
    //             [1.00, 1.00, 0.00, 0.00, 0.00],
    //             [1.00, 1.00, 1.00, 0.00, 0.00],
    //             [1.00, 1.00, 1.00, 1.00, 0.00],
    //             [1.00, 1.00, 1.00, 1.00, 1.00],
    //             [1.00, 1.00, 1.00, 1.00, 1.00],
    //         ]
    //         .into_iter()
    //         .flatten()
    //         .collect::<Vec<_>>(),
    //     );
    // }

    // #[test]
    // fn test_triu() {
    //     let mut cx = Graph::new();

    //     let a = cx.triu(3, -1).retrieve();
    //     let b = cx.triu(3, 0).retrieve();
    //     let c = cx.triu(3, 1).retrieve();

    //     cx.execute();

    //     assert_exact(
    //         &a.data(),
    //         &[[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [0.00, 1.00, 1.00]]
    //             .into_iter()
    //             .flatten()
    //             .collect::<Vec<_>>(),
    //     );
    //     assert_exact(
    //         &b.data(),
    //         &[[1.00, 1.00, 1.00], [0.00, 1.00, 1.00], [0.00, 0.00, 1.00]]
    //             .into_iter()
    //             .flatten()
    //             .collect::<Vec<_>>(),
    //     );
    //     assert_exact(
    //         &c.data(),
    //         &[[0.00, 1.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 0.00]]
    //             .into_iter()
    //             .flatten()
    //             .collect::<Vec<_>>(),
    //     );
    // }
}
