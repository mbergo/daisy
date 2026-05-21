//! Attention mechanisms: scaled dot-product attention, multi-head attention,
//! and the point-wise feed-forward network.

use candle_core::{Result, Tensor, D};
use candle_nn::{linear, Dropout, Linear, Module, VarBuilder};

// ── Scaled dot-product attention ──────────────────────────────────────────────

/// Standard scaled dot-product attention.
///
/// Returns `(output, attention_weights)` where both have the same leading batch
/// and head dimensions as the inputs.
///
/// Mirrors:
/// ```python
/// def scaled_dot_product_attention(q, k, v, mask):
///     ...
/// ```
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let dk = k.dim(D::Minus1)? as f64;
    // Shape: (..., seq_q, seq_k)
    let matmul_qk = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scaled = (matmul_qk / dk.sqrt())?;

    let scaled = match mask {
        Some(m) => {
            // additive mask: large negative value where mask == 1
            let bias = (m * -1e9_f64)?;
            scaled.add(&bias)?
        }
        None => scaled,
    };

    let attention_weights = candle_nn::ops::softmax(&scaled, D::Minus1)?;
    let output = attention_weights.matmul(v)?;
    Ok((output, attention_weights))
}

// ── Multi-head attention ───────────────────────────────────────────────────────

/// Multi-head attention layer.
///
/// Splits `d_model` into `num_heads` heads, projects Q/K/V, applies
/// `scaled_dot_product_attention` per head, then re-projects.
///
/// Mirrors the `MultiHeadAttention` Keras layer in `NPL/npl.py`.
pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    depth: usize,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    dense: Linear,
}

impl MultiHeadAttention {
    /// Creates a new `MultiHeadAttention` layer.
    ///
    /// # Panics
    /// Panics if `d_model % num_heads != 0`.
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let depth = d_model / num_heads;
        let wq = linear(d_model, d_model, vb.pp("wq"))?;
        let wk = linear(d_model, d_model, vb.pp("wk"))?;
        let wv = linear(d_model, d_model, vb.pp("wv"))?;
        let dense = linear(d_model, d_model, vb.pp("dense"))?;
        Ok(Self {
            num_heads,
            d_model,
            depth,
            wq,
            wk,
            wv,
            dense,
        })
    }

    /// Reshapes `[batch, seq, d_model]` → `[batch, num_heads, seq, depth]`.
    fn split_heads(&self, x: &Tensor, batch_size: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        // [batch, seq, d_model] → [batch, seq, num_heads, depth]
        let x = x.reshape((batch_size, seq_len, self.num_heads, self.depth))?;
        // → [batch, num_heads, seq, depth]
        x.transpose(1, 2)?.contiguous()
    }

    /// Forward pass: `(v, k, q, mask)` → `(output, attention_weights)`.
    ///
    /// Argument order mirrors the Python `call(self, v, k, q, mask)`.
    pub fn forward(
        &self,
        v: &Tensor,
        k: &Tensor,
        q: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = q.dim(0)?;

        let q = self.wq.forward(q)?;
        let k = self.wk.forward(k)?;
        let v = self.wv.forward(v)?;

        let q = self.split_heads(&q, batch_size)?;
        let k = self.split_heads(&k, batch_size)?;
        let v = self.split_heads(&v, batch_size)?;

        let (scaled_attention, attention_weights) =
            scaled_dot_product_attention(&q, &k, &v, mask)?;

        // [batch, num_heads, seq, depth] → [batch, seq, num_heads, depth]
        let scaled_attention = scaled_attention.transpose(1, 2)?.contiguous()?;
        let seq_len = scaled_attention.dim(1)?;
        // [batch, seq, d_model]
        let concat_attention =
            scaled_attention.reshape((batch_size, seq_len, self.d_model))?;

        let output = self.dense.forward(&concat_attention)?;
        Ok((output, attention_weights))
    }
}

// ── Point-wise feed-forward network ───────────────────────────────────────────

/// Two-layer point-wise feed-forward network: `Dense(dff, relu) → Dense(d_model)`.
///
/// Mirrors `point_wise_feed_forward_network(d_model, dff)` from `NPL/npl.py`.
pub struct FeedForward {
    dense1: Linear,
    dense2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, dff: usize, vb: VarBuilder) -> Result<Self> {
        let dense1 = linear(d_model, dff, vb.pp("dense1"))?;
        let dense2 = linear(dff, d_model, vb.pp("dense2"))?;
        Ok(Self { dense1, dense2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense1.forward(x)?.relu()?;
        self.dense2.forward(&x)
    }
}

// ── Dropout helper ─────────────────────────────────────────────────────────────

/// Wraps `candle_nn::Dropout` with a training flag, matching the Keras signature
/// `dropout(x, training=training)`.
pub(crate) fn apply_dropout(dropout: &Dropout, x: &Tensor, training: bool) -> Result<Tensor> {
    dropout.forward(x, training)
}
