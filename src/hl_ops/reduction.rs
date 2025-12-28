use crate::{
    op::{self},
    prelude::*,
};

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Sum reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self
                .graph()
                .add_op(op::SumReduce {
                    dim: axes[dim],
                    ..Default::default()
                })
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Max reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self
                .graph()
                .add_op(op::MaxReduce {
                    dim: axes[dim],
                    ..Default::default()
                })
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean(self, axes: impl ToAxes) -> GraphTensor {
        let reduced_elements = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        self.sum(axes) / reduced_elements
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that axis.
    pub fn prod(self, axes: impl ToAxes) -> GraphTensor {
        self.log().sum(axes).exp()
    }
}

#[cfg(test)]
mod tests {
    use crate::hl_ops::unary::tests::test_unary;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_sum() {
        test_unary((2, 3), |a| a.sum(1), |a| a.sum(1).unwrap());
        test_unary((2, 3, 4), |a| a.sum((0, 2)), |a| a.sum((0, 2)).unwrap());
    }

    #[test]
    fn test_max() {
        test_unary((2, 3), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean() {
        test_unary((2, 3), |a| a.mean(1), |a| a.mean(1).unwrap());
        test_unary(
            (2, 3, 4),
            |a| a.mean((0, 2)),
            |a| (a.sum(2).unwrap().sum(0).unwrap() / 8.0).unwrap(),
        );
    }

    #[test]
    fn test_prod() {
        test_unary(
            (2, 3),
            |a| a.prod(1),
            |a| {
                let v = a.to_vec2::<f32>().unwrap();
                let out: Vec<f32> = v.iter().map(|row| row.iter().product()).collect();
                Tensor::from_vec(out, v.len(), &Device::Cpu).unwrap()
            },
        );
    }
}
