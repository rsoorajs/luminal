use luminal::prelude::*;
use rand::rng;

/// A simple layer norm with an optional weight and bias
#[derive(Default)]
pub struct LayerNorm {
    pub weight: Option<GraphTensor>,
    pub bias: Option<GraphTensor>,
    mean_norm: bool,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(
        dim: usize,
        weight: Option<&str>,
        bias: Option<&str>,
        mean_norm: bool,
        epsilon: f32,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: if let Some(w) = weight {
                Some(cx.named_tensor(w, dim))
            } else {
                None
            },
            bias: if let Some(b) = bias {
                Some(cx.named_tensor(b, dim))
            } else {
                None
            },
            mean_norm,
            epsilon,
        }
    }
    // pub fn initialize(self) -> Self {
    //     // Init weight as uniform(-1, 1)
    //     let mut rng = rng();
    //     if let Some(w) = self.weight {
    //         w.set(random_vec_rng(
    //             w.shape.n_elements().to_usize().unwrap(),
    //             &mut rng,
    //         ));
    //     }
    //     if let Some(b) = self.bias {
    //         b.set(random_vec_rng(
    //             b.shape.n_elements().to_usize().unwrap(),
    //             &mut rng,
    //         ));
    //     }
    //     self
    // }
}

impl LayerNorm {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        if self.mean_norm {
            input = input.mean_norm(input.shape.last_axis());
        }
        input = input.std_norm(input.shape.last_axis(), self.epsilon);
        if let Some(w) = self.weight {
            input *= w.expand(input.shape);
        }
        if let Some(b) = self.bias {
            input += b.expand(input.shape);
        }
        input
    }
}
