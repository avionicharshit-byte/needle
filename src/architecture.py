import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn


def default_init():
    return jinit.normal(stddev=0.02)


def residual_init(num_layers):
    return jinit.normal(stddev=0.02 / math.sqrt(2 * num_layers))


DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


class ZCRMSNorm(nn.Module):
    """Zero-centred RMSNorm: scale initialized to 0, applied as (1 + γ) * x / RMS(x)."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(self.dtype)


@dataclass
class TransformerConfig:
    vocab_size: int = 16384
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 2048
    pad_token_id: int = 0
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"
    activation: str = "swiglu"
    no_feedforward: bool = True
    # flash: fused attention via jax.nn.dot_product_attention (auto backend:
    # cudnn flash on H100, XLA elsewhere). No (B,H,T,T) score materialization.
    flash: bool = True
    # remat: per-layer gradient checkpointing. Only needed when activations
    # exceed memory (naive attention at long seq_len, or much larger models).
    remat: bool = False

    def __init__(self, **kwargs):
        valid = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self, k, v)

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len).astype(jnp.float32)
    angles = jnp.outer(t, freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    # rotate in f32 (cos/sin), cast back so q/k/v share a dtype — keeps the
    # fused attention kernel on its bf16 fast path
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class MultiHeadAttention(nn.Module):
    """Self-attention with GQA, QK-norm, and RoPE."""
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        head_dim = self.d_model // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        B = x.shape[0]

        q = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="q_proj")(x)
        k = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="k_proj")(x)
        v = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="v_proj")(x)

        q = q.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.flash:
            impl = "cudnn" if jax.default_backend() == "gpu" else None
            out = jax.nn.dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                mask=mask,
                implementation=impl,
            )
            out = out.reshape(B, -1, self.d_model)
        else:
            repeats = self.num_heads // self.num_kv_heads
            if repeats > 1:
                k = jnp.repeat(k, repeats, axis=1)
                v = jnp.repeat(v, repeats, axis=1)

            scale = jnp.sqrt(jnp.float32(head_dim))
            attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

            attn_weights = nn.softmax(attn_weights, axis=-1)

            out = jnp.matmul(attn_weights, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, -1, self.d_model)

        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="out_proj")(out)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "swiglu"

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="gate_proj")(x)
        up = nn.Dense(self.d_ff, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="up_proj")(x)
        if self.activation == "swiglu":
            h = nn.silu(gate) * up
        elif self.activation == "geglu":
            h = nn.gelu(gate) * up
        else:  # drelu
            h = nn.relu(gate) * nn.relu(up)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="down_proj")(h)


class Block(nn.Module):
    """Pre-norm self-attention (+ optional pre-norm FFN) with gated residuals.

    Gates are scalar sigmoid(g) with g init 0, so every sublayer starts at
    half-strength residual and can learn to sharpen (g→∞) or self-prune (g→-∞).
    """
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "swiglu"
    no_feedforward: bool = True
    flash: bool = True

    @nn.compact
    def __call__(self, x, mask=None, rope=None):
        attn_gate = nn.sigmoid(self.param("attn_gate", jinit.zeros, ())).astype(self.dtype)
        residual = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers, self.dtype, self.flash, name="self_attn")(
            x, mask=mask, rope=rope
        )
        x = residual + attn_gate * x

        if not self.no_feedforward:
            ffn_gate = nn.sigmoid(self.param("ffn_gate", jinit.zeros, ())).astype(self.dtype)
            residual = x
            x = ZCRMSNorm(dtype=self.dtype)(x)
            x = FeedForward(self.d_model, self.d_ff, self.num_layers, self.dtype, self.activation)(x)
            x = residual + ffn_gate * x

        return x


class _ScanBody(nn.Module):
    """Wraps Block for nn.scan: carry = (x, mask, rope).

    The explicit name="block" is load-bearing: with variable_axes={"params": 0}
    the stacked per-layer params land at params["layers"]["block"][...], e.g.
    attn_gate with shape (num_layers,) — read directly for gate logging.
    """
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_ff: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    activation: str = "swiglu"
    no_feedforward: bool = True
    flash: bool = True
    collect_hidden: bool = False

    @nn.compact
    def __call__(self, carry, _):
        x, mask, rope = carry
        x = Block(
            self.num_heads, self.num_kv_heads, self.d_model, self.d_ff,
            self.num_layers, self.dtype, self.activation, self.no_feedforward,
            self.flash, name="block",
        )(x, mask=mask, rope=rope)
        return (x, mask, rope), (x if self.collect_hidden else None)


class Stack(nn.Module):
    """Scanned stack of Blocks + final norm. Returns (x, per_layer_hidden_or_None)."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, collect_hidden=False):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        body = nn.remat(_ScanBody) if cfg.remat else _ScanBody
        ScanBlock = nn.scan(
            body,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=cfg.num_layers,
        )
        (x, _, _), hidden = ScanBlock(
            cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.d_ff,
            cfg.num_layers, dt, cfg.activation, cfg.no_feedforward,
            cfg.flash, collect_hidden, name="layers",
        )((x, mask, rope), None)

        x = ZCRMSNorm(dtype=dt, name="final_norm")(x)
        return x, hidden


class SimpleAttentionNetwork(nn.Module):
    """Decoder-only transformer with tied embeddings, RoPE, GQA, gated residuals,
    and an optional FFN (no_feedforward=True gives the attention-only SAN;
    no_feedforward=False gives the standard-transformer control arm)."""
    config: TransformerConfig

    def setup(self):
        self.embedding = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(self.config.d_model)
        self.stack = Stack(self.config)

    def _rope(self, seq_len):
        head_dim = self.config.d_model // self.config.num_heads
        return precompute_rope_freqs(head_dim, seq_len, self.config.rope_theta)

    def __call__(self, tokens, mask=None):
        """tokens: (B, T) int32. mask: (B, 1, T, T) bool or None (defaults to causal).
        Returns float32 logits (B, T, vocab)."""
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(tokens.shape[1])
        x, _ = self.stack(x, mask=mask, rope=rope)
        return x.astype(jnp.float32) @ self.embedding.embedding.T

    def hidden_states(self, tokens, mask=None):
        """Per-layer post-block hidden states, shape (num_layers, B, T, d_model),
        float32. Materializes all layers — call only on small batches (rank
        diagnostics at eval cadence), never inside the pmapped train step."""
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(tokens.shape[1])
        _, hidden = self.stack(x, mask=mask, rope=rope, collect_hidden=True)
        return hidden.astype(jnp.float32)


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


def make_causal_packing_mask(seg_ids):
    """Block-diagonal causal mask from segment IDs.

    seg_ids: (B, T) int array where 0 = padding, 1+ = document number.
    Returns: (B, 1, T, T) bool mask — token i attends to token j iff j <= i
    and both share the same non-zero segment ID.
    """
    T = seg_ids.shape[1]
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    block = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] > 0)
    return (block & causal[None, :, :])[:, None, :, :]
