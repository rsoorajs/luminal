use luminal::prelude::*;

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (Old weight inputs, Gradient inputs, New weight outputs, Optimizer Graph, Learning Rate Tensor)
pub fn sgd(
    grads: &[(NodeIndex, ShapeTracker)],
) -> (
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Graph,
    GraphTensor,
) {
    let mut opt_graph = Graph::new();
    let (old_weights, gradients): (Vec<NodeIndex>, Vec<NodeIndex>) = grads
        .iter()
        .map(|_| (opt_graph.tensor(1).id, opt_graph.tensor(1).id))
        .unzip();

    let (new_weights, lr) = sgd_on_graph(
        &mut opt_graph,
        &old_weights,
        &gradients
            .iter()
            .zip(grads)
            .map(|(a, (_, b))| (*a, *b))
            .collect::<Vec<_>>(),
    );
    (old_weights, gradients, new_weights, opt_graph, lr)
}

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (New weight outputs, Learning Rate Tensor)
pub fn sgd_on_graph(
    graph: &mut Graph,
    old_weights: impl ToIds,
    grads: &[(NodeIndex, ShapeTracker)],
) -> (Vec<NodeIndex>, GraphTensor) {
    let lr = graph.named_tensor("Learning Rate", 1).set(3e-4).keep(); // Karpathy constant
    let mut new_weights = vec![];
    for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weights.to_ids()) {
        let old_weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
        let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);

        // SGD
        let new_weight = old_weight - (gradient * lr.expand(grad_shape));
        new_weight.keep();

        new_weights.push(new_weight.id);
    }

    (new_weights, lr)
}

struct TensorRef {
    id: NodeIndex,
    shape: ShapeTracker,
}

impl TensorRef {
    fn to_tensor(&self, graph: &mut Graph) -> GraphTensor {
        GraphTensor::from_id(self.id, self.shape, graph)
    }
}

impl From<GraphTensor> for TensorRef {
    fn from(tensor: GraphTensor) -> Self {
        TensorRef {
            id: tensor.id,
            shape: tensor.shape,
        }
    }
}

pub struct AdamOptimizer {
    states: Vec<AdamGradientState>,
    time_input: TensorRef,
    time_output: TensorRef,
    learning_rate: TensorRef,
    beta1: TensorRef,
    beta2: TensorRef,
    epsilon: TensorRef,
}

struct AdamGradientState {
    weight: TensorRef,
    momentum_input: TensorRef,
    momentum_output: TensorRef,
    variance_input: TensorRef,
    variance_output: TensorRef,
}

/// Implements the [Adam](https://arxiv.org/abs/1412.6980) algorithm.
impl AdamOptimizer {
    pub fn new(graph: &mut Graph, old_weights: impl ToIds, grads: &[(NodeIndex, ShapeTracker)]) -> Self {
        let mut states: Vec<AdamGradientState> = Vec::new(); // Placeholder for old weights

        let lr = graph.named_tensor("Learning Rate", 1).set(1e-3).keep();
        let beta1 = graph.named_tensor("Beta1", 1).set(0.9).keep();
        let beta2 = graph.named_tensor("Beta2", 1).set(0.999).keep();
        let epsilon = graph.named_tensor("Epsilon", 1).set(1e-8).keep();
        let one = graph.constant(1.0);
        
        let time_input = graph.tensor(1).set(0.0);
        let time_output = time_input + graph.constant(1.0).expand(time_input.shape);
        time_output.keep();

        for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weights.to_ids()) {
            let shape = grad_shape;
            let weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
            let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);
            let momentum_input = graph.tensor(1).set(0.0).expand(grad_shape).keep();
            let variance_input = graph.tensor(1).set(0.0).expand(grad_shape).keep();

            // Define the momentum update: m = beta1 * m_prev + (1 - beta1) * gradient
            let one_minus_beta1 = one.expand(shape) - beta1.expand(shape);
            let momentum_output = beta1.expand(shape) * momentum_input + one_minus_beta1 * gradient;
            momentum_output.keep(); 

            // Define the variance update: v = beta2 * v_prev + (1 - beta2) * gradient^2
            let one_minus_beta2 = one.expand(shape) - beta2.expand(shape);
            let gradient_squared = gradient * gradient;
            let variance_output = beta2.expand(shape) * variance_input + one_minus_beta2 * gradient_squared;
            variance_output.keep(); 

            let bias_correction1 = one.expand(shape) - beta1.expand(shape).pow(time_output.expand(shape));
            let bias_correction2 = one.expand(shape) - beta2.expand(shape).pow(time_output.expand(shape));

            let m_hat = momentum_output / bias_correction1;
            let v_hat = variance_output / bias_correction2;

            // Adam update: new_weight = old_weight - lr * m_hat / (sqrt(v_hat) + epsilon)
            let denominator = v_hat.sqrt() + epsilon.expand(shape);
            let update = lr.expand(shape) * m_hat / denominator;
            let new_weight = weight - update;
            new_weight.keep();

            let state = AdamGradientState {
                weight: new_weight.into(),
                momentum_input: momentum_input.into(),
                momentum_output: momentum_output.into(),
                variance_input: variance_input.into(),
                variance_output: variance_output.into(),
            };

            states.push(state);
        }

        Self {
            learning_rate: lr.into(),
            time_input: time_input.into(),
            time_output: time_output.into(),
            beta1: beta1.into(),
            beta2: beta2.into(),
            epsilon: epsilon.into(),
            states
        }
    }

    pub fn step_after_execution(&mut self, graph: &mut Graph) {
        let time_input = self.time_input.to_tensor(graph);
        let time_output = self.time_output.to_tensor(graph);

        transfer_data_same_graph(time_output,   time_input, graph);

        for state in &mut self.states {
            let momentum_input = state.momentum_input.to_tensor(graph);
            let momentum_output = state.momentum_output.to_tensor(graph);
            let variance_input = state.variance_input.to_tensor(graph);
            let variance_output = state.variance_output.to_tensor(graph);

            // Update momentum and variance inputs for the next iteration
            transfer_data_same_graph(momentum_output, momentum_input, graph);
            transfer_data_same_graph(variance_output, variance_input, graph);
        }
    }

    pub fn new_weights(&self) -> Vec<NodeIndex> {
        self.states.iter().map(|s| s.weight.id).collect()
    }

    pub fn new_weight_datas(&self, graph: &mut Graph) -> Vec<Vec<f32>> {
        self.states.iter().map(|s| s.weight.to_tensor(graph).data()).collect()
    }

    pub fn time(&self, graph: &mut Graph) -> Vec<f32> {
        let time_input = self.time_input.to_tensor(graph);
        time_input.data()
    }

    pub fn set_beta1(&self, value: f32, graph: &mut Graph) {
        let beta1 = self.beta1.to_tensor(graph);
        beta1.set(value);
    }

    pub fn set_beta2(&self, value: f32, graph: &mut Graph) {
        let beta2 = self.beta2.to_tensor(graph);
        beta2.set(value);
    }

    pub fn set_learning_rate(&self, value: f32, graph: &mut Graph) {
        let lr = self.learning_rate.to_tensor(graph);
        lr.set(value);
    }

    pub fn set_epsilon(&self, value: f32, graph: &mut Graph) {
        let epsilon = self.epsilon.to_tensor(graph);
        epsilon.set(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual validation test with known values
    /// We'll use simple values that are easy to calculate by hand
    #[test]
    fn test_adam_manual_calculation() {
        let mut graph = Graph::new();

        let weights = graph.tensor(1).set(1.0).keep();
        let params = vec![weights.id];
        let gradients = graph.tensor(1).set(0.1).keep();
        let grads = vec![(gradients.id, gradients.shape)]; 

        // Create Adam optimizer
        let adam = AdamOptimizer::new(&mut graph, params, &grads);
        adam.set_beta1(0.9, &mut graph);
        adam.set_beta2(0.999, &mut graph);
        adam.set_learning_rate(1e-3, &mut graph);
        adam.set_epsilon(1e-8, &mut graph);

        graph.compile(GenericCompiler::default(), ());

        // Manual calculation for first step:
        // β₁ = 0.9, β₂ = 0.999, lr = 1e-3, ε = 1e-8
        // m₁ = 0.9 * 0 + 0.1 * 0.1 = 0.01
        // v₁ = 0.999 * 0 + 0.001 * 0.01 = 0.00001
        // m̂₁ = 0.01 / (1 - 0.9¹) = 0.01 / 0.1 = 0.1
        // v̂₁ = 0.00001 / (1 - 0.999¹) = 0.00001 / 0.001 = 0.01
        // update = 1e-3 * 0.1 / (√0.01 + 1e-8) = 1e-4 / 0.1 = 1e-3
        // new_weight = 1.0 - 1e-3 = 0.999
        
        graph.execute();

        let new_weights = adam.new_weight_datas(&mut graph);
        println!("Manual calculation expected: weight = 0.999");
        println!("Actual weight after Adam update: {:?}", new_weights);

        assert_eq!(new_weights[0][0], 0.999, "Weight did not match expected value");

    }

    /// Test Adam on simple quadratic function: f(x) = (x - 3)²
    /// Gradient: f'(x) = 2(x - 3)
    /// Optimal: x* = 3
    #[test] 
    fn test_quadratic_convergence() {
        let mut graph = Graph::new();

        let mut x = graph.tensor(1).set(0.0).keep();
        let target = graph.constant(3.0).keep();
        let two = graph.constant(2.0).keep();

        let mut gradient = two * (x - target);
        gradient.keep();
        let grads = vec![(gradient.id, gradient.shape)];
        let params = vec![x.id];

        let mut adam = AdamOptimizer::new(&mut graph, params, &grads);
        adam.set_beta1(0.9, &mut graph);
        adam.set_beta2(0.999, &mut graph);
        // higher learning rate for faster convergence
        adam.set_learning_rate(1e-2, &mut graph);
        adam.set_epsilon(1e-8, &mut graph);

        graph.compile(GenericCompiler::default(), (&mut x, &mut gradient));
        // Should converge to x ≈ 3.0 after sufficient iterations
        
        for step in 0..1001 {
            graph.execute();

            transfer_data_same_graph(adam.new_weights(), &x, &mut graph);
            adam.step_after_execution(&mut graph);

            if step % 50 == 0 {
                println!("Step {}, Time: {:?}: x = {:?}, target = {:?}, gradients = {:?}", step, adam.time(&mut graph), x.data(), target.data(), gradient.data());
            }

            gradient.drop();
        }

        assert!((x.data()[0] - target.data()[0]).abs() < 1e-3, "Failed to converge to optimum");
    }

    /// Test on Rosenbrock function
    /// f(x,y) = (a-x)² + b(y-x²)²  where a=1, b=100
    /// dx = -2(a-x) - 4b*x(y-x²)
    /// dy = 2b(y-x²)
    /// Optimal: (x*,y*) = (1,1) 
    #[test]
    fn test_rosenbrock_convergence() {
        let mut graph = Graph::new();

        let mut x = graph.tensor(1).set(-1.0).keep();
        let mut y = graph.tensor(1).set(1.0).keep();
        let a = graph.constant(1.0).keep();
        let b = graph.constant(100.0).keep();

        let mut gradient_x = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
        let mut gradient_y = 2.0 * b * (y - x * x);
        gradient_x.keep();
        gradient_y.keep();

        let grads = vec![(gradient_x.id, gradient_x.shape), (gradient_y.id, gradient_y.shape)];
        let params = vec![x.id, y.id];

        let mut adam = AdamOptimizer::new(&mut graph, params, &grads);
        adam.set_beta1(0.9, &mut graph);
        adam.set_beta2(0.999, &mut graph);
        // higher learning rate for faster convergence
        adam.set_learning_rate(1e-2, &mut graph);
        adam.set_epsilon(1e-8, &mut graph);

        graph.compile(GenericCompiler::default(), (&mut x, &mut y, &mut gradient_x, &mut gradient_y));
        // Should converge to x ≈ 3.0 after sufficient iterations

        for step in 0..5000 {
            graph.execute();

            let new_weights = adam.new_weights();
            transfer_data_same_graph(new_weights[0], &x, &mut graph);
            transfer_data_same_graph(new_weights[1], &y, &mut graph);

            adam.step_after_execution(&mut graph);

            if step % 50 == 0 {
                println!("Step {}, Time: {:?}: x = {:?}, y = {:?}, gradient_x = {:?}, gradient_y = {:?}", step, adam.time(&mut graph), x.data(), y.data(), gradient_x.data(), gradient_y.data());
            }

            gradient_x.drop();
            gradient_y.drop();
        }
        
        // Should converge close to (1,1)
        assert!((x.data()[0] - 1.0).abs() < 1e-2, "x didn't converge");
        assert!((y.data()[0] - 1.0).abs() < 1e-2, "y didn't converge");
    }    
}