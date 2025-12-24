use crate::{
    op::{self},
    prelude::*,
};

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Sum reduce each dimension
        for dim in axes.to_axes().into_iter().rev() {
            id = self
                .graph()
                .add_op(op::SumReduce {
                    dim,
                    ..Default::default()
                })
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Max reduce each dimension
        for dim in axes.to_axes().into_iter().rev() {
            id = self
                .graph()
                .add_op(op::MaxReduce {
                    dim,
                    ..Default::default()
                })
                .input(id, 0, shape)
                .finish();
            shape.remove_dim(dim);
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

    #[test]
    fn test_sum() {
        test_unary((2, 3), |a| a.sum(1), |a| a.sum(1).unwrap());
    }

    #[test]
    fn test_max() {
        test_unary((2, 3), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean() {
        test_unary((2, 3), |a| a.mean(1), |a| a.mean(1).unwrap());
    }
}
