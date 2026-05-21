//! Sinusoidal positional encoding helpers.
//!
//! Mirrors `get_angles` and `positional_encoding` from `NPL/npl.py`.

use candle_core::{Device, Result, Tensor};

/// Computes the angle value for position `pos` and embedding dimension `i`
/// in a model of width `d_model`.
///
/// Matches the Python:
/// ```python
/// angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
/// return pos * angle_rates
/// ```
pub fn get_angles(pos: usize, i: usize, d_model: usize) -> f32 {
    let angle_rate = 1.0_f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
    pos as f32 * angle_rate
}

/// Returns the sinusoidal positional encoding tensor of shape `[1, position, d_model]`.
///
/// Even indices receive `sin`, odd indices receive `cos`, matching the Python:
/// ```python
/// angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
/// angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
/// ```
pub fn positional_encoding(position: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0_f32; position * d_model];

    // Fill raw angle values.
    for pos in 0..position {
        for i in 0..d_model {
            data[pos * d_model + i] = get_angles(pos, i, d_model);
        }
    }

    // Apply sin to even indices, cos to odd indices.
    for pos in 0..position {
        for even_i in (0..d_model).step_by(2) {
            data[pos * d_model + even_i] = data[pos * d_model + even_i].sin();
        }
        for odd_i in (1..d_model).step_by(2) {
            data[pos * d_model + odd_i] = data[pos * d_model + odd_i].cos();
        }
    }

    Tensor::from_vec(data, (1, position, d_model), device)
}
