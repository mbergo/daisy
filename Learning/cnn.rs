extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3, Array4, Axis};
use ndarray_rand::{RandomExt, F32};
use rand::distributions::Uniform;

fn main() {
    let input_data = Array::random((1, 1, 28, 28), F32(Uniform::new(0.0, 1.0)));
    let cnn = CNN::new(1, 8, (3, 3), (2, 2));
    let output_data = cnn.forward(&input_data);
    println!("Output data: \n{:?}", output_data);
}

struct CNN {
    conv: Array4<f32>,
    pool_size: (usize, usize),
}

impl CNN {
    pub fn new(input_channels: usize, output_channels: usize, kernel_size: (usize, usize), pool_size: (usize, usize)) -> Self {
        let conv = Array::random((output_channels, input_channels, kernel_size.0, kernel_size.1), F32(Uniform::new(-0.5, 0.5)));
        CNN { conv, pool_size }
    }

    pub fn forward(&self, input: &Array4<f32>) -> Array2<f32> {
        let mut x = convolve(input, &self.conv);
        x.mapv_inplace(relu);
        x = max_pool(&x, self.pool_size);
        flatten(x)
    }
}

fn convolve(input: &Array4<f32>, kernel: &Array4<f32>) -> Array4<f32> {
    let (n, c, h, w) = input.dim();
    let (k_n, k_c, k_h, k_w) = kernel.dim();
    let mut result = Array4::<f32>::zeros((n, k_n, h - k_h + 1, w - k_w + 1));
    for i in 0..k_n {
        for y in 0..h - k_h + 1 {
            for x in 0..w - k_w + 1 {
                let patch = input.slice(s![.., .., y..y + k_h, x..x + k_w]);
                let conv = patch * &kernel.slice(s![i, .., .., ..]);
                result.slice_mut(s![.., i, y, x]).assign(&conv.sum_axis(Axis(1)).sum_axis(Axis(1)).sum_axis(Axis(1)));
            }
        }
    }
    result
}

fn max_pool(input: &Array4<f32>, pool_size: (usize, usize)) -> Array4<f32> {
    let (n, c, h, w) = input.dim();
    let (pool_h, pool_w) = pool_size;
    let mut result = Array4::<f32>::zeros((n, c, h / pool_h, w / pool_w));
    for y in 0..h / pool_h {
        for x in 0..w / pool_w {
            let patch = input
