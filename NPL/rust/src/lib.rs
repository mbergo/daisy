//! `daisy-npl` — Rust port of the Python Transformer Encoder in `NPL/npl.py`.
//!
//! # Modules
//! * [`positional`] — sinusoidal positional encoding helpers
//! * [`attention`]  — scaled dot-product attention, multi-head attention, feed-forward network
//! * [`encoder`]    — `EncoderLayer` and `Encoder` structs

pub mod attention;
pub mod encoder;
pub mod positional;

pub use attention::{scaled_dot_product_attention, FeedForward, MultiHeadAttention};
pub use encoder::{Encoder, EncoderLayer};
pub use positional::{get_angles, positional_encoding};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// Smoke test: build a tiny Encoder and check that the output shape is correct.
    #[test]
    fn test_encoder_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let encoder = Encoder::new(
            2,   // num_layers
            16,  // d_model
            2,   // num_heads
            32,  // dff
            100, // input_vocab_size
            50,  // max_position_encoding
            0.1, // dropout rate
            vb,
        )
        .expect("failed to build Encoder");

        // Dummy token tensor: batch=2, seq_len=8, values in [0, 100).
        let tokens = Tensor::zeros((2_usize, 8_usize), DType::U32, &device)
            .expect("failed to create token tensor");

        let output = encoder
            .forward(&tokens, false, None)
            .expect("forward pass failed");

        assert_eq!(
            output.dims(),
            &[2, 8, 16],
            "expected output shape [2, 8, 16], got {:?}",
            output.dims()
        );
    }
}
