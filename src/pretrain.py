"""Streaming decoder-only pretraining on packed causal-LM blocks.

Streams any registered HF dataset (default PleIAs/SYNTH), packs documents into
fixed-length rows with segment IDs, and trains with doc-boundary-masked CE +
z-loss. Architecture arms (--ffn/--no-ffn) and optimizer arms (--optimizer
muon|adamw) share this single entry point.

Single-node data parallelism via pmap over all local devices (e.g. 8x H100).

Usage:
    san pretrain --wandb
    san pretrain --ffn --optimizer adamw --wandb   # control arm
"""

import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .tokenizer import get_tokenizer
from .data import (
    PrefetchStream,
    build_val_set,
    packed_block_stream,
    resolve_dataset,
)
from .architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    make_causal_packing_mask,
)
from .run import CHECKPOINT_FORMAT_VERSION
from .optim import create_train_state
from .distributed import _replicate, _unreplicate, shard_batch, _upload_checkpoint

_HF_CHECKPOINT_REPO = "Cactus-Compute/checkpoints"


def _losses(apply_fn, params, tokens, seg_ids):
    """Doc-boundary-masked CE + z-loss over one packed block. Returns (total, ce)."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    seg_in, seg_tgt = seg_ids[:, :-1], seg_ids[:, 1:]
    mask = make_causal_packing_mask(seg_in)
    # mask the EOS -> next-doc-BOS step (and any padding, though rows are full)
    loss_mask = ((seg_in == seg_tgt) & (seg_in > 0)).astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(loss_mask), 1.0)

    logits = apply_fn({"params": params}, inputs, mask=mask)
    ce = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits, targets) * loss_mask
    ) / denom
    z_loss = 1e-4 * jnp.sum(jax.nn.logsumexp(logits, axis=-1) ** 2 * loss_mask) / denom
    return ce + z_loss, ce


def _train_step(state, tokens, seg_ids):
    def loss_fn(params):
        return _losses(state.apply_fn, params, tokens, seg_ids)

    (loss, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    ce = jax.lax.pmean(ce, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    return state, loss, ce


def _val_step(state, tokens, seg_ids):
    """Per-device masked NLL sums, psum'd to global totals."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    seg_in, seg_tgt = seg_ids[:, :-1], seg_ids[:, 1:]
    mask = make_causal_packing_mask(seg_in)
    loss_mask = ((seg_in == seg_tgt) & (seg_in > 0)).astype(jnp.float32)

    logits = state.apply_fn({"params": state.params}, inputs, mask=mask)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    nll_sum = jax.lax.psum(jnp.sum(nll * loss_mask), axis_name="batch")
    count = jax.lax.psum(jnp.sum(loss_mask), axis_name="batch")
    return nll_sum, count


def _run_val(p_val_step, state, val_tokens, val_segs, num_devices):
    total_nll, total_count = 0.0, 0.0
    for tokens, segs in zip(val_tokens, val_segs):
        tk = shard_batch(tokens, num_devices)
        sg = shard_batch(segs, num_devices)
        nll, count = p_val_step(state, tk, sg)
        total_nll += float(nll[0])
        total_count += float(count[0])
    loss = total_nll / max(total_count, 1.0)
    return loss, math.exp(min(loss, 20))


def _gate_metrics(params):
    """Per-layer sigmoid(gate) values — a host-side param read, essentially free."""
    block = params["stack"]["layers"]["block"]
    metrics = {}
    attn = np.asarray(jax.nn.sigmoid(block["attn_gate"].astype(jnp.float32)))
    for i, g in enumerate(attn):
        metrics[f"gates/attn_layer_{i}"] = float(g)
    metrics["gates/attn_mean"] = float(attn.mean())
    if "ffn_gate" in block:
        ffn = np.asarray(jax.nn.sigmoid(block["ffn_gate"].astype(jnp.float32)))
        for i, g in enumerate(ffn):
            metrics[f"gates/ffn_layer_{i}"] = float(g)
        metrics["gates/ffn_mean"] = float(ffn.mean())
    return metrics


def _rank_metrics(model, params, tokens, segs, max_rows=2, max_len=257):
    """Effective rank (SVD entropy) of per-layer hidden states on a small val slice.

    Tests the Dong et al. rank-collapse prediction. Materializes (L, B, T, D) —
    keep the slice small, and never call inside the pmapped step.
    """
    tokens = jnp.asarray(tokens[:max_rows, :max_len])
    segs = jnp.asarray(segs[:max_rows, :max_len])
    hidden = model.apply(
        {"params": params}, tokens,
        mask=make_causal_packing_mask(segs),
        method="hidden_states",
    )
    metrics = {}
    for i, layer in enumerate(np.asarray(hidden)):
        X = layer.reshape(-1, layer.shape[-1]).astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        p = s / max(s.sum(), 1e-12)
        p = p[p > 0]
        metrics[f"rank/layer_{i}"] = float(np.exp(-(p * np.log(p)).sum()))
    return metrics


def pretrain(args):
    num_devices = jax.local_device_count()

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(project="san", name=args.name, config=vars(args))

    print(f"\n[1/4] Detecting devices...")
    print(f"      {num_devices} device(s): {jax.devices()[0].device_kind}")

    print(f"\n[2/4] Loading tokenizer and dataset spec...")
    tokenizer = get_tokenizer()
    spec = resolve_dataset(args.dataset, getattr(args, "text_field", None))

    # If resuming, download checkpoint from HF if not local, then adopt its config
    resume_path = getattr(args, "checkpoint", None)
    ckpt_data = None
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"  Checkpoint not found locally, downloading from HF...", flush=True)
            from huggingface_hub import hf_hub_download
            local_dir = os.path.dirname(resume_path) or "checkpoints"
            os.makedirs(local_dir, exist_ok=True)
            resume_path = hf_hub_download(
                repo_id=_HF_CHECKPOINT_REPO,
                filename=os.path.basename(resume_path),
                repo_type="model",
                local_dir=local_dir,
            )
        with open(resume_path, "rb") as f:
            ckpt_data = pickle.load(f)
        version = ckpt_data.get("format_version") if isinstance(ckpt_data, dict) else None
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"{resume_path} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint "
                f"(got {version!r}) — old enc-dec checkpoints cannot be resumed."
            )
        config = TransformerConfig(**ckpt_data["config"])
        print(f"  Config from ckpt: d={config.d_model}, {config.num_layers}L, "
              f"ffn={not config.no_feedforward}", flush=True)
    else:
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_kv_heads=getattr(args, "num_kv_heads", None) or args.num_heads,
            num_layers=args.num_layers,
            d_ff=getattr(args, "d_ff", None) or args.d_model * 4,
            max_seq_len=args.seq_len,
            dtype=args.dtype,
            activation=args.activation,
            no_feedforward=not args.ffn,
        )

    assert config.vocab_size == tokenizer.vocab_size, (
        f"Config vocab {config.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
    )

    global_batch_size = args.batch_size * num_devices
    seq_len = args.seq_len

    total_steps = args.max_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scaled_lr = args.lr * num_devices
    muon_lr = args.muon_lr * math.sqrt(num_devices)

    print(f"\n[3/4] Building val set ({args.val_blocks} blocks)...")
    val_tokens, val_segs, val_docs = build_val_set(
        tokenizer, spec, args.val_blocks, global_batch_size, seq_len
    )
    print(f"      {val_docs:,} docs held out from the stream head")

    print(f"\n[4/4] Initializing model...")
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng, config, scaled_lr, muon_lr, total_steps, warmup_steps,
        args.decay_ratio, optimizer=args.optimizer,
    )

    resume_step = 0
    if ckpt_data is not None:
        # fp16 on disk -> f32 for training (bf16 compute dtype is per-module)
        ckpt_params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), ckpt_data["params"])
        state = state.replace(params=ckpt_params)
        del ckpt_params
        manual_step = getattr(args, "resume_step", None)
        resume_step = manual_step if manual_step is not None else ckpt_data.get("step", 0)
        prev_total = (ckpt_data.get("run") or {}).get("max_steps")
        if prev_total and prev_total != total_steps:
            print(f"  WARNING: resuming with --max-steps {total_steps} but the run "
                  f"started with {prev_total} — the WSD schedule shape will change.",
                  flush=True)
        print(f"  Resumed params from {resume_path} at step {resume_step}", flush=True)

    state = _replicate(state)
    param_count = sum(x.size for x in jax.tree.leaves(_unreplicate(state).params))

    p_train_step = jax.pmap(_train_step, axis_name="batch", donate_argnums=(0,))
    p_val_step = jax.pmap(_val_step, axis_name="batch")

    model = SimpleAttentionNetwork(config)  # host-side apply for rank diagnostics

    run_meta = {
        "optimizer": args.optimizer,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_steps": total_steps,
        "no_feedforward": config.no_feedforward,
    }
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    decay_steps = max(1, int(total_steps * args.decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps
    arch = "attention-only (SAN)" if config.no_feedforward else f"FFN d_ff={config.d_ff}"
    print(f"\n  ─────────────────────────────────────")
    print(f"  Pretraining on {spec.repo}")
    print(f"  ─────────────────────────────────────")
    print(f"  Parameters    {param_count:>12,}")
    print(f"  Architecture  {arch:>12}")
    print(f"  d_model       {config.d_model:>12}")
    print(f"  Heads         {config.num_heads:>7} ({config.num_kv_heads} KV)")
    print(f"  Layers        {config.num_layers:>12}")
    print(f"  Seq len       {seq_len:>12}")
    print(f"  Dtype         {config.dtype:>12}")
    print(f"  Optimizer     {args.optimizer:>12}")
    print(f"  ─────────────────────────────────────")
    print(f"  Devices       {num_devices:>12}")
    print(f"  Batch         {args.batch_size:>7} x {num_devices} = {global_batch_size}")
    print(f"  Adam LR       {args.lr:>7} x {num_devices} = {scaled_lr}")
    print(f"  Muon LR       {args.muon_lr:>7.4f} -> {muon_lr:.4f}")
    print(f"  Schedule      {warmup_steps}w / {stable_steps}s / {decay_steps}d (WSD)")
    print(f"  Total steps   {total_steps:>12,}")
    print(f"  ─────────────────────────────────────\n")

    # Fresh data ordering on resume (avoid re-seeing pre-crash examples)
    stream_seed = args.seed + resume_step
    batch_stream = PrefetchStream(
        lambda: packed_block_stream(tokenizer, spec, global_batch_size, seq_len,
                                    seed=stream_seed, skip_docs=val_docs),
        prefetch=8,
    )

    global_step = resume_step
    pbar = tqdm(desc="Pretrain", total=total_steps, initial=resume_step)

    for tokens, segs in batch_stream:
        if global_step >= total_steps:
            break
        t0 = time.perf_counter()

        tokens_b = shard_batch(tokens, num_devices)
        segs_b = shard_batch(segs, num_devices)
        state, loss, ce = p_train_step(state, tokens_b, segs_b)

        loss_val = float(loss[0])
        ce_val = float(ce[0])
        ppl = math.exp(min(ce_val, 20))
        dt = time.perf_counter() - t0
        global_step += 1

        pbar.update(1)
        pbar.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{ppl:.2f}",
                         tok_s=f"{global_batch_size * seq_len / dt:.0f}")

        if use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss_val,
                "train/ce": ce_val,
                "train/ppl": ppl,
                "train/tokens_per_sec": global_batch_size * seq_len / dt,
                "train/step": global_step,
            }, step=global_step)

        if global_step % args.eval_every == 0:
            val_loss, val_ppl = _run_val(p_val_step, state, val_tokens, val_segs, num_devices)
            host_params = _unreplicate(state).params
            metrics = {"val/loss": val_loss, "val/ppl": val_ppl}
            metrics.update(_gate_metrics(host_params))
            log_rank_every = getattr(args, "log_rank_every", 0)
            if log_rank_every and global_step % log_rank_every == 0:
                metrics.update(_rank_metrics(model, host_params,
                                             val_tokens[0], val_segs[0]))
            pbar.write(f"  [step {global_step}] val loss {val_loss:.4f}, ppl {val_ppl:.2f}")
            if use_wandb:
                import wandb
                wandb.log(metrics, step=global_step)

        if global_step % args.save_every == 0:
            _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, global_step,
                             upload=getattr(args, "upload_checkpoints", False))

    batch_stream.close()
    pbar.close()

    ckpt_path = _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, global_step,
                                 upload=getattr(args, "upload_checkpoints", False))
    print(f"\nPretraining complete. {global_step} steps.")
    print(f"Checkpoint: {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


def _save_checkpoint(state, config, run_meta, checkpoint_dir, name, global_step, upload=False):
    """Save <name>.pkl (format v2), optionally uploading to HF hub."""
    params = _unreplicate(state).params
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)

    ckpt_path = os.path.join(checkpoint_dir, f"{name}.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "params": params_np,
            "config": config.__dict__,
            "step": global_step,
            "run": run_meta,
        }, f)

    param_count = sum(x.size for x in jax.tree.leaves(params_np))
    size_mb = sum(x.nbytes for x in jax.tree.leaves(params_np)) / 1e6
    print(f"\n  [step {global_step}] Saved {ckpt_path} ({param_count:,} params, {size_mb:.1f} MB)")

    if upload:
        _upload_checkpoint(ckpt_path)
    return ckpt_path
