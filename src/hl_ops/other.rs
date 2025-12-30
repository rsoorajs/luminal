use crate::{op::Constant, prelude::*};

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
            self.add_op(Iota(
                i.into().simplify(),
                sh.iter().copied().product::<Expression>().simplify(),
            ))
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

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size);
        let vertical = self.arange(size).expand_dim(1, size);
        (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size);
        let vertical = self.arange(size).expand_dim(1, size);
        (horizontal - (diagonal as f32 - 1.)).gt(vertical)
    }
}

impl GraphTensor {
    pub fn cast(self, dtype: DType) -> GraphTensor {
        if self.dtype == dtype {
            return self;
        }
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
    use crate::{prelude::*, tests::assert_close};
    use candle_core::{Device, Tensor};

    pub fn test_init(
        func: impl Fn(&mut Graph) -> GraphTensor,
        ref_func: impl Fn(&Device) -> Tensor,
    ) {
        let mut cx = Graph::new();
        let b = func(&mut cx).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = ref_func(&device).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
    }

    #[test]
    fn test_arange() {
        test_init(
            |cx| cx.arange(13).cast(DType::F32) * 1.0,
            |dev| Tensor::arange(0_f32, 13_f32, dev).unwrap(),
        );
        test_init(
            |cx| cx.arange_options(-5, 25, 5).cast(DType::F32) * 1.0,
            |dev| {
                (Tensor::arange(-1_f32, 5_f32, dev).unwrap()
                    * Tensor::new(5_f32, dev).unwrap().broadcast_as(6).unwrap())
                .unwrap()
            },
        );
        test_init(
            |cx| cx.arange_options(0, 4, 1).cast(DType::F32) / 3.,
            #[allow(clippy::excessive_precision)]
            |dev| Tensor::new(vec![0_f32, 0.3333333333, 0.666666666, 0.99999999], dev).unwrap(),
        );
    }

    #[test]
    fn test_gather() {
        test_init(
            |cx| {
                cx.arange(13)
                    .cast(DType::F32)
                    .gather(cx.iota(Expression::from('z') * 2, 5))
            },
            |dev| Tensor::new(vec![0_f32, 2., 4., 6., 8.], dev).unwrap(),
        );
    }

    #[test]
    fn test_triangle_mask() {
        test_init(
            |cx| cx.tril(10, 0).cast(DType::F32),
            |dev| Tensor::tril2(10, candle_core::DType::F32, dev).unwrap(),
        );
        test_init(
            |cx| cx.triu(43, 0).cast(DType::F32),
            |dev| Tensor::triu2(43, candle_core::DType::F32, dev).unwrap(),
        );
    }
}
