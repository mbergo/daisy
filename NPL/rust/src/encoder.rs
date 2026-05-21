//! Transformer encoder layer and full encoder stack.

use candle_core::{Result, Tensor};
use candle_nn::{embedding, layer_norm, Dropout, Embedding, LayerNorm, LayerNormConfig, Module, VarBuilder};

use crate::attention::{apply_dropout, FeedForward, MultiHeadAttention};
use crate::positional::positional_encoding;

// ── EncoderLayer ───────────────────────────────────────────────────────────────

/// A single Transformer encoder layer.
///
/// Architecture: MHA → dropout → residual + LayerNorm → FFN → dropout → residual + LayerNorm.
///
/// Mirrors the `EncoderLayer` Keras layer in `NPL/npl.py` with the syntax error
/// on the double-assignment corrected to a single dropout application.
pub struct EncoderLayer {
    mha: MultiHeadAttention,
    ffn: FeedForward,
    layernorm1: LayerNorm,
    layernorm2: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
}

impl EncoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dff: usize,
        rate: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mha = MultiHeadAttention::new(d_model, num_heads, vb.pp("mha"))?;
        let ffn = FeedForward::new(d_model, dff, vb.pp("ffn"))?;
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let layernorm1 = layer_norm(d_model, norm_cfg, vb.pp("layernorm1"))?;
        let layernorm2 = layer_norm(d_model, norm_cfg, vb.pp("layernorm2"))?;
        let dropout1 = Dropout::new(rate);
        let dropout2 = Dropout::new(rate);
        Ok(Self {
            mha,
            ffn,
            layernorm1,
            layernorm2,
            dropout1,
            dropout2,
        })
    }

    /// Forward pass.
    ///
    /// * `x`        — input tensor `[batch, seq, d_model]`
    /// * `training` — whether to apply dropout
    /// * `mask`     — optional attention mask
    pub fn forward(
        &self,
        x: &Tensor,
        training: bool,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention sub-layer.
        let (attn_output, _) = self.mha.forward(x, x, x, mask)?;
        let attn_output = apply_dropout(&self.dropout1, &attn_output, training)?;
        let out1 = self.layernorm1.forward(&x.add(&attn_output)?)?;

        // Feed-forward sub-layer.
        let ffn_output = self.ffn.forward(&out1)?;
        let ffn_output = apply_dropout(&self.dropout2, &ffn_output, training)?;
        let out2 = self.layernorm2.forward(&out1.add(&ffn_output)?)?;

        Ok(out2)
    }
}

// ── Encoder ────────────────────────────────────────────────────────────────────

/// Full Transformer encoder stack.
///
/// Consists of a token embedding, sinusoidal positional encoding, input dropout,
/// and a stack of `num_layers` `EncoderLayer`s.
///
/// Mirrors the `Encoder` Keras layer in `NPL/npl.py`.
pub struct Encoder {
    d_model: usize,
    embedding: Embedding,
    pos_encoding: Tensor,
    enc_layers: Vec<EncoderLayer>,
    dropout: Dropout,
}

impl Encoder {
    /// Creates a new `Encoder`.
    ///
    /// # Arguments
    /// * `num_layers`             — number of encoder layers
    /// * `d_model`                — model dimension
    /// * `num_heads`              — number of attention heads (must divide `d_model`)
    /// * `dff`                    — inner dimension of the feed-forward network
    /// * `input_vocab_size`       — vocabulary size for the token embedding
    /// * `max_position_encoding`  — maximum sequence length for positional encoding
    /// * `rate`                   — dropout rate
    /// * `vb`                     — `VarBuilder` used to allocate all trainable parameters
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        dff: usize,
        input_vocab_size: usize,
        max_position_encoding: usize,
        rate: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let emb = embedding(input_vocab_size, d_model, vb.pp("embedding"))?;
        let pos_encoding = positional_encoding(max_position_encoding, d_model, vb.device())?;
        let enc_layers = (0..num_layers)
            .map(|i| {
                EncoderLayer::new(
                    d_model,
                    num_heads,
                    dff,
                    rate,
                    vb.pp(format!("layer_{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let dropout = Dropout::new(rate);
        Ok(Self {
            d_model,
            embedding: emb,
            pos_encoding,
            enc_layers,
            dropout,
        })
    }

    /// Forward pass.
    ///
    /// * `x`        — integer token tensor `[batch, seq_len]` (dtype `U32`)
    /// * `training` — whether to apply dropout
    /// * `mask`     — optional attention mask
    ///
    /// Returns a float tensor of shape `[batch, seq_len, d_model]`.
    pub fn forward(
        &self,
        x: &Tensor,
        training: bool,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = x.dim(1)?;

        // Token embedding + scale by sqrt(d_model).
        let mut x = self.embedding.forward(x)?;
        x = (x * (self.d_model as f64).sqrt())?;

        // Add positional encoding (broadcast over batch dimension).
        let pos = self.pos_encoding.narrow(1, 0, seq_len)?;
        x = x.broadcast_add(&pos)?;

        // Input dropout.
        x = apply_dropout(&self.dropout, &x, training)?;

        // Encoder layers.
        for layer in &self.enc_layers {
            x = layer.forward(&x, training, mask)?;
        }

        Ok(x)
    }
}
