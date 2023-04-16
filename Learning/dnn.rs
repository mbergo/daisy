extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Axis};
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Uniform;

fn main() {
    let input_data = Array::random((100, 4), F32(Uniform::new(0.0, 1.0)));
    let dnn = DNN::new(&[4, 8, 4, 1]);
    let output_data = dnn.forward(&input_data);
    println!("Output data: \n{:?}", output_data);
}

struct DNN {
    layers: Vec<Array2<f32>>,
}

impl DNN {
    pub fn new(sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Array::random((sizes[i], sizes[i + 1]), F32(Uniform::new(-0.5, 0.5))));
        }
        DNN { layers }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x = input.clone();
        for i in 0..self.layers.len() {
            x = x.dot(&self.layers[i]);
            if i < self.layers.len() - 1 {
                x.mapv_inplace(relu);
            }
        }
        x
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}
