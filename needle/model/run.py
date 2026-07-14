import pickle
import sys

import jax
import jax.numpy as jnp

from ..dataset.tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from .architecture import SimpleAttentionNetwork, TransformerConfig

CHECKPOINT_FORMAT_VERSION = 2

_decode_fn_cache = {}


def load_checkpoint(path):
    """Load a format-v2 checkpoint. Returns (params, config)."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    version = ckpt.get("format_version") if isinstance(ckpt, dict) else None
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint "
            f"(got format_version={version!r}). Old encoder-decoder/tool-calling "
            f"checkpoints are incompatible with this branch."
        )
    config = ckpt["config"]
    if isinstance(config, dict):
        config = TransformerConfig(**config)
    return ckpt["params"], config


def _get_decode_fn(model, buf_len):
    """Cached jit forward over a fixed-size token buffer (full recompute per step).

    Fine for qualitative checks at ≤26M params; a KV cache is future work.
    """
    key = (id(model), buf_len)
    if key not in _decode_fn_cache:
        @jax.jit
        def decode_fn(params, tokens):
            return model.apply({"params": params}, tokens)
        _decode_fn_cache[key] = decode_fn
    return _decode_fn_cache[key]


def generate(model, params, tokenizer, prompt, max_new_tokens=256, temperature=0.0,
             seed=0, stream=True):
    """Greedy (temperature=0) or sampled continuation of a prompt. Returns the
    generated text (prompt excluded)."""
    prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
    buf_len = min(model.config.max_seq_len, len(prompt_ids) + max_new_tokens)
    if len(prompt_ids) >= buf_len:
        raise ValueError(f"Prompt ({len(prompt_ids)} tokens) does not fit in max_seq_len={model.config.max_seq_len}")

    buffer = jnp.full((1, buf_len), PAD_ID, dtype=jnp.int32)
    buffer = buffer.at[0, :len(prompt_ids)].set(jnp.array(prompt_ids, dtype=jnp.int32))
    decode_fn = _get_decode_fn(model, buf_len)
    rng = jax.random.PRNGKey(seed)

    generated = []
    printed = ""
    for pos in range(len(prompt_ids) - 1, buf_len - 1):
        logits = decode_fn(params, buffer)[0, pos]
        if temperature <= 0.0:
            next_token = int(jnp.argmax(logits))
        else:
            rng, sample_rng = jax.random.split(rng)
            next_token = int(jax.random.categorical(sample_rng, logits / temperature))
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        buffer = buffer.at[0, pos + 1].set(next_token)
        if stream:
            text = tokenizer.decode(generated)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text

    text = tokenizer.decode(generated)
    if stream:
        sys.stdout.write(text[len(printed):] + "\n")
        sys.stdout.flush()
    return text


def main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    prompt = args.prompt or "The most surprising thing about"
    print(f"prompt: {prompt!r}")
    generate(
        model, params, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
