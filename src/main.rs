extern crate ndarray;
extern crate rand;

use ndarray::prelude::*;
use rand::Rng;

#[derive(Debug, Clone)]
enum ActivationFunction {
    Sigmoid,
    ReLU,
}

struct NeuralNetwork {
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
    activation_function: ActivationFunction,
    weights: Vec<Array2<f64>>, // Use Array2<f64> for weights
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize, activation_function: ActivationFunction) -> Self {
        let mut weights = Vec::new();
        let input_dim = input_size;
        let output_dim = output_size;
        let layer_sizes: Vec<usize> = [input_dim]
            .iter()
            .chain(hidden_sizes.iter())
            .chain(&[output_dim])
            .cloned() // Clone the values into the Vec
            .collect();
        
        for i in 0..layer_sizes.len() - 1 {
            let (input_dim, output_dim) = (layer_sizes[i], layer_sizes[i + 1]);
            let weight_matrix = Array::from_shape_fn((input_dim, output_dim), |_| rand::thread_rng().gen_range(-1.0..1.0));
            weights.push(weight_matrix);
        }
        
        NeuralNetwork {
            input_size,
            hidden_sizes,
            output_size,
            activation_function,
            weights,
        }
    }
    
    fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        
        for weight_matrix in &self.weights {
            let layer_input = output.dot(weight_matrix);
            output = match self.activation_function {
                ActivationFunction::Sigmoid => sigmoid(layer_input),
                ActivationFunction::ReLU => relu(layer_input),
            };
        }
        
        output
    }
}

fn sigmoid(x: Array1<f64>) -> Array1<f64> {
    1.0 / (1.0 + (-x).mapv(f64::exp))
}

fn relu(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

fn main() {
    let input_size = 3;
    let hidden_sizes = vec![3, 4]; // Specify the sizes of hidden layers
    let output_size = 1;
    let activation_function = ActivationFunction::Sigmoid; // Change the activation function here

    let neural_network = NeuralNetwork::new(input_size, hidden_sizes, output_size, activation_function);
    
    let input_data = arr1(&[0.5, 0.8,0.4]);
    let prediction = neural_network.predict(&input_data);
    
    println!("Prediction: {:?}", prediction);
}
