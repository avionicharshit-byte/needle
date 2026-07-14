import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .tokenizer import get_tokenizer
from .pretraining import resolve_dataset, build_val_set
from .architecture import SimpleAttentionNetwork, make_causal_packing_mask
from .run import load_checkpoint, generate

_val_loss_fn_cache = {}


def _get_val_loss_fn(model):
    key = id(model)
    if key not in _val_loss_fn_cache:
        @jax.jit
        def batch_nll(params, tokens, segs):
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            seg_in, seg_tgt = segs[:, :-1], segs[:, 1:]
            mask = make_causal_packing_mask(seg_in)
            logits = model.apply({"params": params}, inputs, mask=mask)
            loss_mask = ((seg_in == seg_tgt) & (seg_in > 0)).astype(jnp.float32)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return jnp.sum(nll * loss_mask), jnp.sum(loss_mask)
        _val_loss_fn_cache[key] = batch_nll
    return _val_loss_fn_cache[key]


def compute_val_loss(model, params, val_tokens, val_segs):
    """Exact masked NLL over packed val blocks (num_blocks, B, T+1).

    Returns (loss_per_token, perplexity). Same doc-boundary masking as training.
    """
    batch_nll = _get_val_loss_fn(model)
    total_nll, total_tokens = 0.0, 0.0
    for tokens, segs in zip(val_tokens, val_segs):
        nll, count = batch_nll(params, jnp.asarray(tokens), jnp.asarray(segs))
        total_nll += float(nll)
        total_tokens += float(count)
    loss = total_nll / max(total_tokens, 1.0)
    return loss, math.exp(min(loss, 20))


def measure_throughput(model, params, tokenizer, num_runs=10,
                       prompt="The history of science shows that", max_new_tokens=64):
    """Decode tokens/sec via the full-recompute sampler (qualitative-scale only)."""
    # warmup (jit compile)
    generate(model, params, tokenizer, prompt, max_new_tokens=8, stream=False)

    total_tokens, total_time = 0, 0.0
    for run in range(num_runs):
        start = time.perf_counter()
        text = generate(model, params, tokenizer, prompt,
                        max_new_tokens=max_new_tokens, temperature=0.8,
                        seed=run, stream=False)
        total_time += time.perf_counter() - start
        total_tokens += len(tokenizer.encode(text))

    return {
        "tokens_per_second": total_tokens / max(total_time, 1e-9),
        "avg_latency_s": total_time / num_runs,
    }


def compute_repetition_rate(texts):
    bigram_rep_rates = []
    for text in texts:
        words = text.lower().split()
        if len(words) < 2:
            bigram_rep_rates.append(0.0)
            continue
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique = len(set(bigrams))
        bigram_rep_rates.append(1.0 - unique / len(bigrams))
    return float(np.mean(bigram_rep_rates))


DEFAULT_SAMPLE_PROMPTS = [
    "The most important discovery in physics was",
    "To solve this problem, first consider",
    "The capital of France is",
    "Once upon a time",
]


def sample_generations(model, params, tokenizer, prompts=None, max_new_tokens=128, temperature=0.8):
    prompts = prompts or DEFAULT_SAMPLE_PROMPTS
    generations = []
    for i, prompt in enumerate(prompts):
        text = generate(model, params, tokenizer, prompt,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        seed=i, stream=False)
        generations.append((prompt, text))
    rep_rate = compute_repetition_rate([g for _, g in generations])
    return {"generations": generations, "bigram_repetition_rate": rep_rate}


def evaluate_downstream(model, params, tokenizer, tasks):
    """Downstream benchmark stub (lm-eval-harness adapter, future work).

    Planned shape: a wrapper exposing loglikelihood(context, continuation)
    implemented as one packed causal forward with the continuation positions
    loss-masked in — the same primitives compute_val_loss uses. Tasks like
    lambada/hellaswag/arc_easy then score via the standard harness.
    """
    if tasks:
        raise NotImplementedError(f"Downstream tasks not implemented yet: {tasks}")
    return {}


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    print(f"Model: {config.num_layers}L d={config.d_model} "
          f"{'no-FFN (SAN)' if config.no_feedforward else f'FFN d_ff={config.d_ff}'}")

    spec = resolve_dataset(args.dataset, getattr(args, "text_field", None))
    seq_len = min(args.seq_len, config.max_seq_len)
    print(f"Building {args.val_blocks} val blocks from {spec.repo} (seq_len={seq_len})...")
    val_tokens, val_segs, _ = build_val_set(tokenizer, spec, args.val_blocks, args.batch_size, seq_len)

    loss, ppl = compute_val_loss(model, params, val_tokens, val_segs)
    print(f"\nVal loss         {loss:>10.4f}")
    print(f"Val perplexity   {ppl:>10.2f}")

    tp = measure_throughput(model, params, tokenizer, num_runs=args.throughput_runs)
    print(f"Throughput       {tp['tokens_per_second']:>10.1f} tok/s")
    print(f"Avg latency      {tp['avg_latency_s']:>10.3f} s")

    samples = sample_generations(model, params, tokenizer)
    print(f"Bigram repetition{samples['bigram_repetition_rate']:>10.1%}\n")
    for prompt, text in samples["generations"]:
        print(f"  > {prompt!r}\n    {text!r}\n")

    evaluate_downstream(model, params, tokenizer, getattr(args, "tasks", None))
