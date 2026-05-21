# daisy-npl — Rust Transformer Encoder

A Rust port of [`NPL/npl.py`](../npl.py), which implements a Transformer **Encoder** stack using TensorFlow/Keras.

## Chosen tensor crate

This port uses [**candle**](https://github.com/huggingface/candle) (`candle-core` + `candle-nn`) by Hugging Face.

**Why candle over burn or tch?**
- Pure Rust, no native C++ / libtorch dependency (unlike `tch`).
- Minimal, low-level API that maps cleanly to the explicit tensor operations in the Python original.
- Stable 0.9 release with full CPU support — no GPU required to build or test.
- `candle-nn` provides ready-made `Linear`, `Embedding`, `LayerNorm`, and `Dropout` primitives that mirror the Keras layers used in the original.

`burn` was considered but its higher-level module system (trait-based `Module` with associated `Config` types) would add more boilerplate than clarity for a direct port of this size.

## Structure

```
NPL/rust/
├── Cargo.toml
├── README.md          ← you are here
└── src/
    ├── lib.rs         — crate root + smoke test
    ├── positional.rs  — get_angles, positional_encoding
    ├── attention.rs   — scaled_dot_product_attention, MultiHeadAttention, FeedForward
    └── encoder.rs     — EncoderLayer, Encoder
```

## What is implemented

| Python | Rust |
|---|---|
| `get_angles(pos, i, d_model)` | `positional::get_angles(pos, i, d_model) -> f32` |
| `positional_encoding(position, d_model)` | `positional::positional_encoding(position, d_model, device) -> Result<Tensor>` |
| `scaled_dot_product_attention(q, k, v, mask)` | `attention::scaled_dot_product_attention(q, k, v, mask) -> Result<(Tensor, Tensor)>` |
| `MultiHeadAttention(d_model, num_heads)` | `attention::MultiHeadAttention::new(d_model, num_heads, vb)` |
| `point_wise_feed_forward_network(d_model, dff)` | `attention::FeedForward::new(d_model, dff, vb)` |
| `EncoderLayer(d_model, num_heads, dff, rate)` | `encoder::EncoderLayer::new(d_model, num_heads, dff, rate, vb)` |
| `Encoder(num_layers, d_model, num_heads, dff, vocab, max_pos, rate)` | `encoder::Encoder::new(num_layers, d_model, num_heads, dff, vocab, max_pos, rate, vb)` |

The double-assignment syntax artifact in the Python source (`attn_output = attn_output = ...`) is corrected to a single assignment in the Rust port.

## Build and test

```bash
cd NPL/rust
cargo build
cargo test
```

The `cargo test` command runs a smoke test that:
1. Builds a small `Encoder` (`num_layers=2, d_model=16, num_heads=2, dff=32, vocab=100, max_pos=50`).
2. Runs a forward pass on a `[batch=2, seq_len=8]` token tensor (no mask, inference mode).
3. Asserts the output shape is `[2, 8, 16]`.
