import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    make_causal_mask,
    make_causal_packing_mask,
)
from src.optim import create_train_state, _param_labels


def tiny_config(**overrides):
    base = dict(
        vocab_size=64, d_model=32, num_heads=4, num_kv_heads=2,
        num_layers=3, d_ff=64, max_seq_len=32, dtype="bfloat16",
        no_feedforward=True,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def init_model(config, seed=0):
    model = SimpleAttentionNetwork(config)
    tokens = jnp.ones((2, 16), dtype=jnp.int32)
    params = model.init({"params": jax.random.PRNGKey(seed)}, tokens)["params"]
    return model, params, tokens


def n_params(params):
    return sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))


def test_forward_shape_and_dtype():
    config = tiny_config()
    model, params, tokens = init_model(config)
    logits = model.apply({"params": params}, tokens)
    assert logits.shape == (2, 16, config.vocab_size)
    assert logits.dtype == jnp.float32
    assert not bool(jnp.isnan(logits).any())


def test_ffn_toggle_param_delta():
    _, params_san, _ = init_model(tiny_config(no_feedforward=True))
    _, params_ffn, _ = init_model(tiny_config(no_feedforward=False))
    cfg = tiny_config()
    # swiglu FFN adds 3 matrices per layer (+ scalar gate + norm scale)
    expected_ffn = cfg.num_layers * (3 * cfg.d_model * cfg.d_ff + 1 + cfg.d_model)
    assert n_params(params_ffn) - n_params(params_san) == expected_ffn


def test_gate_param_paths():
    cfg = tiny_config(no_feedforward=False)
    _, params, _ = init_model(cfg)
    block = params["stack"]["layers"]["block"]
    assert block["attn_gate"].shape == (cfg.num_layers,)
    assert block["ffn_gate"].shape == (cfg.num_layers,)
    # gates init to 0 -> sigmoid = 0.5 (half-strength residual)
    assert np.allclose(jax.nn.sigmoid(block["attn_gate"]), 0.5)

    _, params_san, _ = init_model(tiny_config(no_feedforward=True))
    assert "ffn_gate" not in params_san["stack"]["layers"]["block"]


def test_hidden_states_shape():
    cfg = tiny_config()
    model, params, tokens = init_model(cfg)
    hidden = model.apply({"params": params}, tokens, method="hidden_states")
    assert hidden.shape == (cfg.num_layers, 2, 16, cfg.d_model)
    assert hidden.dtype == jnp.float32


def test_causal_packing_mask():
    seg = jnp.array([[1, 1, 2, 2, 0]])
    mask = make_causal_packing_mask(seg)[0, 0]
    expected = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],  # padding attends to nothing
    ], dtype=bool)
    assert np.array_equal(np.asarray(mask), expected)


def test_packed_forward_matches_separate():
    """Two docs packed in one row give the same logits as run separately."""
    cfg = tiny_config(dtype="float32")
    model, params, _ = init_model(cfg)
    doc_a = jnp.array([[3, 4, 5]], dtype=jnp.int32)
    doc_b = jnp.array([[6, 7]], dtype=jnp.int32)
    packed = jnp.array([[3, 4, 5, 6, 7]], dtype=jnp.int32)
    seg = jnp.array([[1, 1, 1, 2, 2]])

    logits_packed = model.apply({"params": params}, packed, mask=make_causal_packing_mask(seg))
    logits_a = model.apply({"params": params}, doc_a)
    logits_b = model.apply({"params": params}, doc_b)

    np.testing.assert_allclose(np.asarray(logits_packed[0, :3]), np.asarray(logits_a[0]), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(logits_packed[0, 3:]), np.asarray(logits_b[0]), rtol=2e-4, atol=2e-4)


def test_causal_mask_no_leakage():
    """Changing a future token must not change past logits."""
    cfg = tiny_config(dtype="float32")
    model, params, _ = init_model(cfg)
    a = jnp.array([[3, 4, 5, 6]], dtype=jnp.int32)
    b = a.at[0, 3].set(9)
    la = model.apply({"params": params}, a)
    lb = model.apply({"params": params}, b)
    np.testing.assert_allclose(np.asarray(la[0, :3]), np.asarray(lb[0, :3]), rtol=1e-5, atol=1e-5)


def test_param_labels_kernels_muon():
    _, params, _ = init_model(tiny_config(no_feedforward=False))
    labels = _param_labels(params)
    flat = jax.tree_util.tree_flatten_with_path(labels)[0]
    for path, label in flat:
        name = path[-1].key
        if name == "kernel":
            assert label == "muon", path
        else:
            assert label == "adam", path
    assert any(label == "muon" for _, label in flat)


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
@pytest.mark.parametrize("no_ffn", [True, False])
def test_train_state_step(optimizer, no_ffn):
    cfg = tiny_config(no_feedforward=no_ffn)
    state = create_train_state(
        jax.random.PRNGKey(0), cfg, learning_rate=1e-3, muon_lr=0.01,
        total_steps=100, warmup_steps=10, optimizer=optimizer,
    )
    tokens = jnp.ones((2, 16), dtype=jnp.int32)
    targets = jnp.ones((2, 16), dtype=jnp.int32)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, tokens)
        import optax
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    # two steps: warmup LR is 0 at step 0, so movement only shows from step 2
    new_state = state
    for _ in range(2):
        loss, grads = jax.value_and_grad(loss_fn)(new_state.params)
        assert bool(jnp.isfinite(loss))
        new_state = new_state.apply_gradients(grads=grads)
    moved = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), state.params, new_state.params)
    assert any(jax.tree.leaves(moved))


def test_rezero_is_identity_at_init():
    """alpha init 0 -> every block is the identity -> logits reduce to
    final_norm(embed) @ embed.T, computable by hand."""
    import math
    cfg = tiny_config(residual="rezero", dtype="float32")
    model, params, tokens = init_model(cfg)
    block = params["stack"]["layers"]["block"]
    assert block["attn_alpha"].shape == (cfg.num_layers,)
    assert "attn_gate" not in block

    logits = model.apply({"params": params}, tokens)
    emb = np.asarray(params["embedding"]["embedding"])
    x = emb[np.asarray(tokens)] * math.sqrt(cfg.d_model)
    rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
    expected = (x / rms) @ emb.T
    np.testing.assert_allclose(np.asarray(logits), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", ["standard", "none"])
def test_unscaled_residual_variants(mode):
    cfg = tiny_config(residual=mode, no_feedforward=False, dtype="float32")
    model, params, tokens = init_model(cfg)
    block = params["stack"]["layers"]["block"]
    for name in ("attn_gate", "attn_alpha", "ffn_gate", "ffn_alpha"):
        assert name not in block
    logits = model.apply({"params": params}, tokens)
    assert bool(jnp.isfinite(logits).all())


def test_rms_norm_matches_zcrms_at_init():
    """γ=1 RMSNorm and γ=0 zero-centered RMSNorm are the same function at init
    (and optimizer-equivalent under kernel-only weight decay)."""
    model_z, params_z, tokens = init_model(tiny_config(dtype="float32"))
    model_r, params_r, _ = init_model(tiny_config(dtype="float32", norm="rms"))
    scale = params_r["stack"]["layers"]["block"]["RMSNorm_0"]["scale"]
    assert np.allclose(np.asarray(scale), 1.0)
    logits_z = model_z.apply({"params": params_z}, tokens)
    logits_r = model_r.apply({"params": params_r}, tokens)
    np.testing.assert_allclose(np.asarray(logits_z), np.asarray(logits_r), rtol=1e-6, atol=1e-6)


def test_qk_norm_toggle():
    _, params, tokens = init_model(tiny_config(qk_norm=False))
    attn = params["stack"]["layers"]["block"]["self_attn"]
    assert "q_norm" not in attn and "k_norm" not in attn
    model, params, tokens = init_model(tiny_config(qk_norm=False))
    logits = model.apply({"params": params}, tokens)
    assert bool(jnp.isfinite(logits).all())


def test_post_attn_norm_params():
    cfg = tiny_config(post_attn_norm=True, no_feedforward=False)
    model, params, tokens = init_model(cfg)
    block = params["stack"]["layers"]["block"]
    assert "post_attn_norm" in block and "post_ffn_norm" in block
    logits = model.apply({"params": params}, tokens)
    assert bool(jnp.isfinite(logits).all())


def test_variant_train_step():
    """Kitchen-sink E3 variant survives a muon step under scan+remat."""
    cfg = tiny_config(residual="rezero", norm="rms", qk_norm=False, post_attn_norm=True)
    state = create_train_state(
        jax.random.PRNGKey(0), cfg, learning_rate=1e-3, muon_lr=0.01,
        total_steps=100, warmup_steps=10, optimizer="muon",
    )
    tokens = jnp.ones((2, 16), dtype=jnp.int32)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, tokens)
        import optax
        return optax.softmax_cross_entropy_with_integer_labels(logits, tokens).mean()

    new_state = state
    for _ in range(2):
        loss, grads = jax.value_and_grad(loss_fn)(new_state.params)
        assert bool(jnp.isfinite(loss))
        new_state = new_state.apply_gradients(grads=grads)
    moved = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), state.params, new_state.params)
    assert any(jax.tree.leaves(moved))


def test_flash_matches_naive():
    """Fused attention path must be numerically equivalent to the naive path."""
    params = init_model(tiny_config(dtype="float32", flash=False))[1]
    tokens = jnp.arange(2 * 16, dtype=jnp.int32).reshape(2, 16) % 64
    logits_naive = SimpleAttentionNetwork(tiny_config(dtype="float32", flash=False)).apply(
        {"params": params}, tokens)
    logits_flash = SimpleAttentionNetwork(tiny_config(dtype="float32", flash=True)).apply(
        {"params": params}, tokens)
    np.testing.assert_allclose(np.asarray(logits_naive), np.asarray(logits_flash), rtol=2e-4, atol=2e-4)
