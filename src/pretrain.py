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
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .tokenizer import HF_REPO, get_tokenizer
from .data import (
    PrefetchStream,
    build_val_set,
    memmap_block_stream,
    memmap_val_set,
    packed_block_stream,
    resolve_dataset,
)
from .architecture import (
    SimpleAttentionNetwork,
    TransformerConfig,
    make_causal_packing_mask,
)
from .run import CHECKPOINT_FORMAT_VERSION
from .optim import _wsd_schedule, create_train_state


# --- single-node pmap plumbing -------------------------------------------

def _replicate(tree):
    """Replicate a pytree across all local devices for pmap."""
    devices = np.array(jax.local_devices())
    n = len(devices)
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def _rep(x):
        x = jnp.asarray(x)
        return jax.device_put(jnp.broadcast_to(x, (n,) + x.shape), sharding)

    return jax.tree.map(_rep, tree)


def _unreplicate(tree):
    """Get a single copy from a pmap-replicated pytree."""
    return jax.tree.map(lambda x: jax.device_get(x[0]), tree)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def _upload_checkpoint(ckpt_path, wait=False):
    """Upload a checkpoint file to HuggingFace Hub in a background thread.

    wait=True blocks until the upload finishes — required for the final
    checkpoint, because daemon threads are killed at interpreter exit and a
    fire-and-forget upload seconds before process end can be truncated.
    """
    import threading

    def _upload():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
            filename = os.path.basename(ckpt_path)
            print(f"[hf] Uploading {filename} to {HF_REPO}/checkpoints/ ...")
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"checkpoints/{filename}",
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"[hf] Checkpoint uploaded: {HF_REPO}/checkpoints/{filename}")
        except Exception as e:
            print(f"[hf] Warning: checkpoint upload failed: {e}")

    t = threading.Thread(target=_upload, daemon=True)
    t.start()
    if wait:
        t.join(timeout=900)


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
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, ce, grad_norm


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
    """Per-layer residual-scale values — a host-side param read, essentially free.

    gated -> sigmoid(gate); rezero -> raw alpha (can be negative);
    standard/none -> no params, empty metrics.
    """
    block = params["stack"]["layers"]["block"]
    metrics = {}
    for kind in ("attn", "ffn"):
        if f"{kind}_gate" in block:
            vals = np.asarray(jax.nn.sigmoid(block[f"{kind}_gate"].astype(jnp.float32)))
        elif f"{kind}_alpha" in block:
            vals = np.asarray(block[f"{kind}_alpha"].astype(jnp.float32))
        else:
            continue
        for i, g in enumerate(vals):
            metrics[f"gates/{kind}_layer_{i}"] = float(g)
        metrics[f"gates/{kind}_mean"] = float(vals.mean())
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
                repo_id=HF_REPO,
                filename=f"checkpoints/{os.path.basename(resume_path)}",
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
            residual=getattr(args, "residual", "gated"),
            norm=getattr(args, "norm", "zcrms"),
            qk_norm=not getattr(args, "no_qk_norm", False),
            post_attn_norm=getattr(args, "post_attn_norm", False),
        )

    assert config.vocab_size == tokenizer.vocab_size, (
        f"Config vocab {config.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
    )

    global_batch_size = args.batch_size * num_devices
    seq_len = args.seq_len

    # E0-locked per-arm LR defaults (experiments.md §11, 2026-07-15); explicit
    # flags override. Muon runs keep the adam side at 3e-4 — as swept in E0.
    is_ffn = not config.no_feedforward
    base_muon_lr = args.muon_lr if args.muon_lr is not None else (0.04 if is_ffn else 0.02)
    if args.lr is not None:
        base_lr = args.lr
    elif args.optimizer == "adamw":
        base_lr = 2.4e-3 if is_ffn else 6e-4
    else:
        base_lr = 3e-4

    total_steps = args.max_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scaled_lr = base_lr * num_devices
    muon_lr = base_muon_lr * math.sqrt(num_devices)
    # host-side copies of the schedules, for logging only
    adam_schedule = _wsd_schedule(scaled_lr, total_steps, warmup_steps, args.decay_ratio)
    muon_schedule = _wsd_schedule(muon_lr, total_steps, warmup_steps, args.decay_ratio)

    data_dir = getattr(args, "data_dir", None)
    print(f"\n[3/4] Building val set ({args.val_blocks} blocks)...")
    if data_dir:
        val_tokens, val_segs = memmap_val_set(data_dir, args.val_blocks,
                                              global_batch_size, seq_len)
        val_docs = 0  # memmap path holds out a fixed val pool by doc index
        print(f"      from memmap corpus at {data_dir} (fixed val pool)")
    else:
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

    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = "unknown"
    run_meta = {
        "optimizer": args.optimizer,
        "dataset": args.dataset,
        "seed": args.seed,
        "max_steps": total_steps,
        "no_feedforward": config.no_feedforward,
        "batch_size": args.batch_size,
        "num_devices": num_devices,
        "lr": scaled_lr,
        "muon_lr": muon_lr,
        "commit": commit,
        "argv": " ".join(sys.argv[1:]),
    }
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Milestone checkpoints for E7 (spectra at 0/25/50/75/100% of training):
    # kept as separate files, never overwritten. Final save covers 100%.
    milestone_steps = sorted({max(1, int(total_steps * f)) for f in (0.25, 0.5, 0.75)})
    if resume_step == 0:
        _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, 0,
                         upload=getattr(args, "upload_checkpoints", False),
                         milestone=True)  # init state = the 0% point

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
    print(f"  Adam LR       {base_lr:>7} x {num_devices} = {scaled_lr}")
    print(f"  Muon LR       {base_muon_lr:>7.4f} -> {muon_lr:.4f}")
    print(f"  Schedule      {warmup_steps}w / {stable_steps}s / {decay_steps}d (WSD)")
    print(f"  Total steps   {total_steps:>12,}")
    print(f"  ─────────────────────────────────────\n")

    # Fresh data ordering on resume (avoid re-seeing pre-crash examples)
    stream_seed = args.seed + resume_step
    if data_dir:
        batch_stream = PrefetchStream(
            lambda: memmap_block_stream(data_dir, global_batch_size, seq_len,
                                        seed=stream_seed,
                                        max_docs=getattr(args, "max_docs", None)),
            prefetch=32,
        )
    else:
        batch_stream = PrefetchStream(
            lambda: packed_block_stream(tokenizer, spec, global_batch_size, seq_len,
                                        seed=stream_seed, skip_docs=val_docs,
                                        max_docs=getattr(args, "max_docs", None)),
            prefetch=32,
        )

    global_step = resume_step
    pbar = tqdm(desc="Pretrain", total=total_steps, initial=resume_step)

    for tokens, segs in batch_stream:
        if global_step >= total_steps:
            break
        t0 = time.perf_counter()

        tokens_b = shard_batch(tokens, num_devices)
        segs_b = shard_batch(segs, num_devices)
        state, loss, ce, grad_norm = p_train_step(state, tokens_b, segs_b)

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
                "train/grad_norm": float(grad_norm[0]),
                "train/lr": float(adam_schedule(global_step)),
                "train/muon_lr": float(muon_schedule(global_step)) if args.optimizer == "muon" else 0.0,
                "train/tokens": global_step * global_batch_size * seq_len,
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
            do_upload = (getattr(args, "upload_checkpoints", False)
                         and global_step % getattr(args, "upload_every", 10_000) == 0)
            _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, global_step,
                             upload=do_upload)
        if global_step in milestone_steps:
            _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, global_step,
                             upload=getattr(args, "upload_checkpoints", False), milestone=True)

    batch_stream.close()
    pbar.close()

    ckpt_path = _save_checkpoint(state, config, run_meta, checkpoint_dir, args.name, global_step,
                                 upload=getattr(args, "upload_checkpoints", False),
                                 upload_wait=True)
    print(f"\nPretraining complete. {global_step} steps.")
    print(f"Checkpoint: {ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


def _save_checkpoint(state, config, run_meta, checkpoint_dir, name, global_step,
                     upload=False, milestone=False, upload_wait=False):
    """Save <name>.pkl (format v2), optionally uploading to HF hub.

    milestone=True writes <name>_step<k>.pkl instead — a kept copy that is
    never overwritten (E7 needs weight snapshots across training).
    """
    params = _unreplicate(state).params
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float16), params)

    fname = f"{name}_step{global_step}.pkl" if milestone else f"{name}.pkl"
    ckpt_path = os.path.join(checkpoint_dir, fname)
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
        _upload_checkpoint(ckpt_path, wait=upload_wait)
    return ckpt_path
