"""Streaming packed causal-LM data pipeline.

Documents are streamed from a HuggingFace dataset, tokenized as
[BOS] + text + [EOS], and packed into fixed-length rows with per-document
segment IDs. Training uses `make_causal_packing_mask(seg_ids)` so attention
never crosses document boundaries. A document that spans a row boundary
continues as segment 1 of the next row (its context truncates at the edge).

Rows have length seq_len + 1; the train step derives inputs = tokens[:, :-1]
and targets = tokens[:, 1:].
"""

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .tokenizer import BOS_ID, EOS_ID


@dataclass
class DatasetSpec:
    repo: str
    fmt: Callable
    split: str = "train"
    name: Optional[str] = None


def _fmt_synth(ex):
    """PleIAs/SYNTH: query + reasoning trace + answer as one document.

    SYNTH is reasoning-by-design — 97%+ of samples carry a synthetic_reasoning
    trace, and the dataset's SOTA-for-size results are for the full format.
    Traces also externalize intermediate computation into context, the regime
    the SAN hypothesis targets.
    """
    parts = [ex.get("query"), ex.get("synthetic_reasoning"), ex.get("synthetic_answer")]
    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    return text or None


def _fmt_text_field(field_name):
    def fmt(ex):
        text = ex.get(field_name)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    return fmt


DATASETS = {
    "synth": DatasetSpec(repo="PleIAs/SYNTH", fmt=_fmt_synth),
    "fineweb-edu": DatasetSpec(repo="HuggingFaceFW/fineweb-edu", name="sample-10BT", fmt=_fmt_text_field("text")),
}


def resolve_dataset(name, text_field=None):
    """Registry lookup, or an ad-hoc spec for any HF repo with a text column."""
    if name in DATASETS:
        return DATASETS[name]
    return DatasetSpec(repo=name, fmt=_fmt_text_field(text_field or "text"))


def stream_texts(spec, seed=42, shuffle=True, skip=0, take=None):
    """Yield formatted document strings. skip/take apply BEFORE shuffle so a
    val holdout taken from the unshuffled head is deterministic and disjoint
    from a train stream that skips it."""
    from datasets import load_dataset

    ds = load_dataset(spec.repo, spec.name, split=spec.split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    if take:
        ds = ds.take(take)
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=1000)

    for example in ds:
        text = spec.fmt(example)
        if text:
            yield text


def _pack_rows(tokenizer, texts, batch_size, seq_len):
    """Pack a text stream into (tokens[B, T+1], seg_ids[B, T+1]) int32 batches.

    Rows are always completely filled, so seg_id 0 (padding) never occurs;
    the final partial row/batch of a finite stream is dropped.
    """
    row_len = seq_len + 1
    rows_t, rows_s = [], []
    cur_t, cur_s = [], []
    seg = 0

    for text in texts:
        ids = [BOS_ID] + tokenizer.encode(text) + [EOS_ID]
        pos = 0
        while pos < len(ids):
            if not cur_t:
                seg = 0
            seg += 1
            space = row_len - len(cur_t)
            chunk = ids[pos:pos + space]
            cur_t.extend(chunk)
            cur_s.extend([seg] * len(chunk))
            pos += len(chunk)
            if len(cur_t) == row_len:
                rows_t.append(np.array(cur_t, dtype=np.int32))
                rows_s.append(np.array(cur_s, dtype=np.int32))
                cur_t, cur_s = [], []
                if len(rows_t) == batch_size:
                    yield np.stack(rows_t), np.stack(rows_s)
                    rows_t, rows_s = [], []


def packed_block_stream(tokenizer, spec, batch_size, seq_len, seed=42, skip_docs=0, max_docs=None):
    """Infinite shuffled train stream of packed batches.

    Cycles the dataset when exhausted, reshuffling with a fresh seed each
    epoch, so training always reaches --max-steps regardless of corpus size.
    max_docs caps the unique documents drawn per epoch (skip/take applied
    before shuffle, so the subset is fixed across epochs) — the knob for
    data-scaling runs.
    """
    epoch = 0
    while True:
        texts = stream_texts(spec, seed=seed + epoch, shuffle=True,
                             skip=skip_docs, take=max_docs)
        yield from _pack_rows(tokenizer, texts, batch_size, seq_len)
        epoch += 1
        print(f"[data] dataset pass {epoch} complete — cycling with fresh shuffle", flush=True)


def build_val_set(tokenizer, spec, num_blocks, batch_size, seq_len, oversample=10, seed=3407):
    """Pack a seeded 1-in-`oversample` subsample of the stream head into the val set.

    Sampling across a window `oversample`x wider than the raw head guards
    against ordering artifacts in the corpus (e.g. amplified samples grouped
    by seed article) while staying deterministic and independent of the
    training shuffle seed.

    Returns (tokens[num_blocks, B, T+1], seg_ids[num_blocks, B, T+1], docs_consumed),
    where docs_consumed counts the FULL window. Pass it as skip_docs to
    packed_block_stream so train never sees any window doc (kept or not).
    """
    import random
    rng = random.Random(seed)
    counter = {"docs": 0}

    def sampled_texts():
        for text in stream_texts(spec, shuffle=False):
            counter["docs"] += 1
            if rng.random() < 1.0 / oversample:
                yield text

    blocks_t, blocks_s = [], []
    for tokens, segs in _pack_rows(tokenizer, sampled_texts(), batch_size, seq_len):
        blocks_t.append(tokens)
        blocks_s.append(segs)
        if len(blocks_t) == num_blocks:
            break
    if len(blocks_t) < num_blocks:
        raise RuntimeError(
            f"Stream exhausted after {len(blocks_t)}/{num_blocks} val blocks — "
            f"reduce --val-blocks or check the dataset."
        )
    return np.stack(blocks_t), np.stack(blocks_s), counter["docs"]


class PrefetchStream:
    """Prefetch streaming batches in a background thread."""

    def __init__(self, generator_fn, prefetch=4):
        self._queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, args=(generator_fn,), daemon=True)
        self._thread.start()

    def _produce(self, gen_fn):
        try:
            for batch in gen_fn():
                if self._stop.is_set():
                    return
                self._queue.put(batch)
            self._queue.put(None)
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop.set()
