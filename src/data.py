"""Streaming packed causal-LM data pipeline.

Documents are streamed from a HuggingFace dataset, tokenized as
[BOS] + text + [EOS], and packed into fixed-length rows with per-document
segment IDs. Training uses `make_causal_packing_mask(seg_ids)` so attention
never crosses document boundaries. A document that spans a row boundary
continues as segment 1 of the next row (its context truncates at the edge).

Rows have length seq_len + 1; the train step derives inputs = tokens[:, :-1]
and targets = tokens[:, 1:].
"""

import io
import json
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .tokenizer import BOS_ID, EOS_ID, IM_START, IM_END, THINK_START, THINK_END


@dataclass
class DatasetSpec:
    repo: str
    fmt: Callable
    split: str = "train"
    name: Optional[str] = None
    # parquet columns the formatter needs (None = read all; used by tokenize_corpus)
    fields: Optional[List[str]] = None


def _fmt_synth(ex):
    """PleIAs/SYNTH as single-turn ChatML with a thinking trace.

    Matches the Monad/Baguettotron format so their published numbers are
    comparable reference points. SYNTH is reasoning-by-design (97%+ of samples
    carry a trace); the markers are atomic tokenizer symbols, so trace vs
    answer regions can be located exactly for per-region loss analysis.
    """
    q = (ex.get("query") or "").strip()
    r = (ex.get("synthetic_reasoning") or "").strip()
    a = (ex.get("synthetic_answer") or "").strip()
    if not q or not a:
        return None
    think = f"{THINK_START}\n{r}\n{THINK_END}\n" if r else ""
    return (f"{IM_START}user\n{q}{IM_END}\n"
            f"{IM_START}assistant\n{think}{a}{IM_END}")


def _fmt_text_field(field_name):
    def fmt(ex):
        text = ex.get(field_name)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    return fmt


DATASETS = {
    "synth": DatasetSpec(repo="PleIAs/SYNTH", fmt=_fmt_synth,
                         fields=["query", "synthetic_reasoning", "synthetic_answer"]),
    "fineweb-edu": DatasetSpec(repo="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                               fmt=_fmt_text_field("text"), fields=["text"]),
}


def resolve_dataset(name, text_field=None):
    """Registry lookup, or an ad-hoc spec for any HF repo with a text column."""
    if name in DATASETS:
        return DATASETS[name]
    return DatasetSpec(repo=name, fmt=_fmt_text_field(text_field or "text"),
                       fields=[text_field or "text"])


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


def _encode_stream(tokenizer, texts, workers=8, ahead=256):
    """Tokenize documents with a sliding window of worker threads.

    SentencePiece's Encode releases the GIL, so threads give real parallelism.
    Results yield in submission order, so output is deterministic regardless
    of thread timing.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        for text in texts:
            pending.append(executor.submit(tokenizer.encode, text))
            if len(pending) >= ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _pack_token_stream(doc_ids_iter, batch_size, seq_len):
    """Pack an iterator of token-id lists (already BOS/EOS-wrapped) into
    (tokens[B, T+1], seg_ids[B, T+1]) int32 batches.

    Rows are always completely filled, so seg_id 0 (padding) never occurs;
    the final partial row/batch of a finite stream is dropped.
    """
    row_len = seq_len + 1
    rows_t, rows_s = [], []
    cur_t, cur_s = [], []
    seg = 0

    for ids in doc_ids_iter:
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


def _pack_rows(tokenizer, texts, batch_size, seq_len):
    """Tokenize a text stream and pack it (see _pack_token_stream)."""
    def docs():
        for doc_ids in _encode_stream(tokenizer, texts):
            yield [BOS_ID] + doc_ids + [EOS_ID]
    yield from _pack_token_stream(docs(), batch_size, seq_len)


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


# ---------------------------------------------------------------------------
# Pre-tokenized memmap corpus: tokenize once, train forever.
#
# Layout in <data_dir>:
#   tokens.bin    uint16 token ids, all docs concatenated (no BOS/EOS)
#   offsets.bin   int64 end-offset per document into tokens.bin
#   manifest.json {"done_files": [...], "docs": N, "tokens": M, ...}
#
# The writer is resumable per parquet shard (crash loses at most one shard's
# work); the reader validates file sizes against the manifest, so a torn
# write from a crash fails loudly instead of being trained on.
# ---------------------------------------------------------------------------

CORPUS_TOKENS = "tokens.bin"
CORPUS_OFFSETS = "offsets.bin"
CORPUS_MANIFEST = "manifest.json"
_CORPUS_BASE_SEED = 3407  # fixed base permutation: defines the val pool
VAL_POOL_DOCS = 100_000


def _write_manifest(data_dir, manifest):
    tmp = os.path.join(data_dir, CORPUS_MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, os.path.join(data_dir, CORPUS_MANIFEST))


def _process_parquet_file(path, spec, tokenizer, bin_f, off_f, base_offset):
    """Tokenize one parquet shard, append to the corpus files.

    Returns (docs_appended, tokens_appended). Deterministic given the shard,
    formatter, and tokenizer. Reads the parquet in bounded batches — never
    materializes a whole shard in memory (OOM-safe on small CPU pods).
    """
    import pyarrow.parquet as pq

    def texts():
        with pq.ParquetFile(path) as pf:
            for batch in pf.iter_batches(batch_size=2048, columns=spec.fields):
                for row in batch.to_pylist():
                    t = spec.fmt(row)
                    if t:
                        yield t

    docs = tokens = 0
    offset = base_offset
    for ids in _encode_stream(tokenizer, texts()):
        arr = np.asarray(ids, dtype=np.uint16)
        bin_f.write(arr.tobytes())
        offset += len(ids)
        tokens += len(ids)
        docs += 1
        off_f.write(np.int64(offset).tobytes())
    return docs, tokens


def tokenize_corpus(dataset="synth", text_field=None, out_dir=None, force=False,
                    max_files=None):
    """One-time corpus tokenization to a uint16 memmap. Resumable: re-running
    the same command continues from the last completed parquet shard."""
    import time as _time
    from huggingface_hub import HfApi, hf_hub_download
    from .tokenizer import get_tokenizer

    spec = resolve_dataset(dataset, text_field)
    tokenizer = get_tokenizer()
    assert tokenizer.vocab_size <= 65535, "uint16 corpus requires vocab <= 65535"

    out_dir = out_dir or os.path.join("data", dataset)
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, CORPUS_TOKENS)
    off_path = os.path.join(out_dir, CORPUS_OFFSETS)
    man_path = os.path.join(out_dir, CORPUS_MANIFEST)

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(spec.repo, repo_type="dataset")
                   if f.endswith(".parquet"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No parquet files found in {spec.repo}")

    manifest = {"repo": spec.repo, "done_files": [], "docs": 0, "tokens": 0}
    if os.path.exists(man_path) and not force:
        with open(man_path) as f:
            manifest = json.load(f)
        print(f"[corpus] resuming: {len(manifest['done_files'])}/{len(files)} shards, "
              f"{manifest['tokens']:,} tokens so far", flush=True)
    elif force:
        for p in (bin_path, off_path, man_path):
            if os.path.exists(p):
                os.remove(p)

    for p, unit, count in ((bin_path, 2, manifest["tokens"]), (off_path, 8, manifest["docs"])):
        open(p, "ab").close()
        if os.path.getsize(p) != count * unit:
            print(f"[corpus] truncating torn tail of {os.path.basename(p)}", flush=True)
            os.truncate(p, count * unit)

    dl_dir = os.path.join(out_dir, "_dl")
    todo = [f for f in files if f not in manifest["done_files"]]

    done_n = len(manifest["done_files"])
    per_shard = manifest["tokens"] * 2 / max(done_n, 1) if done_n else 280e6
    need = per_shard * len(todo) + 1e9
    free = os.statvfs(out_dir).f_frsize * os.statvfs(out_dir).f_bavail
    if free < need:
        raise RuntimeError(
            f"~{need / 1e9:.0f} GB still needed at {out_dir} but only "
            f"{free / 1e9:.0f} GB free — grow the volume before starting "
            f"(progress so far is preserved; re-run to resume)."
        )

    t0 = _time.time()
    session_start_tokens = manifest["tokens"]  # rate must exclude resumed work
    with open(bin_path, "ab") as bin_f, open(off_path, "ab") as off_f:
        for i, fname in enumerate(todo):
            bin_pos, off_pos = bin_f.tell(), off_f.tell()
            for attempt in range(4):
                try:
                    local = hf_hub_download(spec.repo, fname, repo_type="dataset",
                                            local_dir=dl_dir)
                    docs, tokens = _process_parquet_file(local, spec, tokenizer,
                                                         bin_f, off_f, manifest["tokens"])
                    break
                except Exception as e:
                    bin_f.flush(); off_f.flush()
                    bin_f.truncate(bin_pos); off_f.truncate(off_pos)
                    if attempt == 3:
                        raise
                    wait = 15 * (attempt + 1)
                    print(f"[corpus] shard {fname} failed ({type(e).__name__}: {e}) — "
                          f"retry {attempt + 1}/3 in {wait}s", flush=True)
                    _time.sleep(wait)
            bin_f.flush(); os.fsync(bin_f.fileno())
            off_f.flush(); os.fsync(off_f.fileno())
            manifest["docs"] += docs
            manifest["tokens"] += tokens
            manifest["done_files"].append(fname)
            _write_manifest(out_dir, manifest)
            try:
                os.remove(local)  # keep peak disk = output + one shard
            except OSError:
                pass
            done = len(manifest["done_files"])
            rate = (manifest["tokens"] - session_start_tokens) / max(_time.time() - t0, 1)
            print(f"[corpus] {done}/{len(files)} shards | {manifest['docs']:,} docs | "
                  f"{manifest['tokens']:,} tokens | ~{rate/1e6:.1f}M tok/s this session",
                  flush=True)

    print(f"[corpus] complete: {manifest['tokens']:,} tokens "
          f"({manifest['tokens'] * 2 / 1e9:.1f} GB) at {out_dir}", flush=True)
    return out_dir


def _load_corpus(data_dir):
    """Open a tokenized corpus, validating sizes against the manifest."""
    man_path = os.path.join(data_dir, CORPUS_MANIFEST)
    with open(man_path) as f:
        manifest = json.load(f)
    bin_path = os.path.join(data_dir, CORPUS_TOKENS)
    off_path = os.path.join(data_dir, CORPUS_OFFSETS)
    if os.path.getsize(bin_path) != manifest["tokens"] * 2 or \
       os.path.getsize(off_path) != manifest["docs"] * 8:
        raise RuntimeError(
            f"Corpus at {data_dir} does not match its manifest (torn write from a "
            f"crash?). Re-run `san tokenize-corpus` — it resumes and repairs."
        )
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    ends = np.fromfile(off_path, dtype=np.int64)
    return tokens, ends


def _corpus_doc(tokens, ends, i):
    start = 0 if i == 0 else int(ends[i - 1])
    return [BOS_ID] + tokens[start:int(ends[i])].tolist() + [EOS_ID]


def _corpus_pools(n_docs):
    """Fixed base permutation → disjoint (val_pool, train_pool) doc indices."""
    base = np.random.default_rng(_CORPUS_BASE_SEED).permutation(n_docs)
    n_val = min(VAL_POOL_DOCS, n_docs // 10)
    return base[:n_val], base[n_val:]


def memmap_val_set(data_dir, num_blocks, batch_size, seq_len):
    """Pack the fixed val pool into (tokens[n,B,T+1], seg_ids[n,B,T+1])."""
    tokens, ends = _load_corpus(data_dir)
    val_pool, _ = _corpus_pools(len(ends))

    def docs():
        for i in val_pool:
            yield _corpus_doc(tokens, ends, int(i))

    blocks_t, blocks_s = [], []
    for t, s in _pack_token_stream(docs(), batch_size, seq_len):
        blocks_t.append(t)
        blocks_s.append(s)
        if len(blocks_t) == num_blocks:
            break
    if len(blocks_t) < num_blocks:
        raise RuntimeError(f"Val pool too small for {num_blocks} blocks — reduce --val-blocks.")
    return np.stack(blocks_t), np.stack(blocks_s)


def memmap_block_stream(data_dir, batch_size, seq_len, seed=42, max_docs=None):
    """Infinite train stream over the memmap corpus: a fresh seeded global
    permutation of the train pool each epoch (val pool always excluded).
    max_docs restricts to a fixed subset of the train pool across epochs."""
    tokens, ends = _load_corpus(data_dir)
    _, train_pool = _corpus_pools(len(ends))
    if max_docs:
        train_pool = train_pool[:max_docs]

    epoch = 0
    while True:
        order = np.random.default_rng(seed + epoch).permutation(train_pool)

        def docs():
            for i in order:
                yield _corpus_doc(tokens, ends, int(i))

        yield from _pack_token_stream(docs(), batch_size, seq_len)
        epoch += 1
        print(f"[data] corpus pass {epoch} complete — cycling with fresh shuffle", flush=True)


# --- corpus <-> HuggingFace Hub (tokens.bin exceeds HF per-file limits, so it
# --- travels as fixed-size parts and is reassembled on download)

CORPUS_HF_PREFIX = "corpus"
CORPUS_PART_BYTES = 10 * 1024**3  # 10 GB parts


class _FileSlice(io.BufferedIOBase):
    """Seekable read-only view of a byte range of a file — lets us upload
    part N of tokens.bin without writing a 10GB copy or buffering it in RAM.

    Inherits io.BufferedIOBase because huggingface_hub validates
    path_or_fileobj with isinstance(..., io.BufferedIOBase)."""

    def __init__(self, path, start, length):
        self._f = open(path, "rb")
        self._start = start
        self._length = length
        self._f.seek(start)

    def read(self, n=-1):
        remaining = self._start + self._length - self._f.tell()
        if remaining <= 0:
            return b""
        return self._f.read(remaining if n is None or n < 0 else min(n, remaining))

    def seek(self, pos, whence=0):
        if whence == 0:
            target = self._start + pos
        elif whence == 1:
            target = self._f.tell() + pos
        else:  # relative to slice end
            target = self._start + self._length + pos
        self._f.seek(min(max(target, self._start), self._start + self._length))
        return self.tell()

    def tell(self):
        return self._f.tell() - self._start

    def seekable(self):
        return True

    def readable(self):
        return True

    def close(self):
        self._f.close()
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def upload_corpus(data_dir, dataset="synth", part_bytes=CORPUS_PART_BYTES):
    """Upload a completed corpus to HF under corpus/<dataset>/. Resumable:
    parts already present in the repo are skipped; the manifest is uploaded
    last, acting as the completeness marker for download_corpus."""
    from huggingface_hub import HfApi
    from .tokenizer import HF_REPO

    _load_corpus(data_dir)  # validates sizes against the manifest first
    with open(os.path.join(data_dir, CORPUS_MANIFEST)) as f:
        manifest = json.load(f)

    bin_path = os.path.join(data_dir, CORPUS_TOKENS)
    size = os.path.getsize(bin_path)
    n_parts = max(1, -(-size // part_bytes))
    prefix = f"{CORPUS_HF_PREFIX}/{dataset}"

    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    existing = set(api.list_repo_files(HF_REPO, repo_type="model"))

    for i in range(n_parts):
        name = f"{prefix}/tokens.bin.part{i:04d}"
        if name in existing:
            print(f"[corpus] part {i + 1}/{n_parts} already uploaded — skipping", flush=True)
            continue
        start = i * part_bytes
        with _FileSlice(bin_path, start, min(part_bytes, size - start)) as fs:
            api.upload_file(path_or_fileobj=fs, path_in_repo=name,
                            repo_id=HF_REPO, repo_type="model")
        print(f"[corpus] uploaded part {i + 1}/{n_parts}", flush=True)

    api.upload_file(path_or_fileobj=os.path.join(data_dir, CORPUS_OFFSETS),
                    path_in_repo=f"{prefix}/{CORPUS_OFFSETS}",
                    repo_id=HF_REPO, repo_type="model")
    manifest = dict(manifest, parts=n_parts, part_bytes=part_bytes)
    api.upload_file(path_or_fileobj=json.dumps(manifest).encode(),
                    path_in_repo=f"{prefix}/{CORPUS_MANIFEST}",
                    repo_id=HF_REPO, repo_type="model")
    print(f"[corpus] upload complete: {HF_REPO}/{prefix} ({size / 1e9:.1f} GB)", flush=True)


def download_corpus(dataset="synth", out_dir=None):
    """Fetch a corpus from HF and reassemble it. Resumable: completed parts
    are not re-downloaded (tokens.bin is truncated to the last part boundary
    and appended from there)."""
    import shutil
    from huggingface_hub import hf_hub_download
    from .tokenizer import HF_REPO

    out_dir = out_dir or os.path.join("data", dataset)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{CORPUS_HF_PREFIX}/{dataset}"
    dl_dir = os.path.join(out_dir, "_dl")

    man_local = hf_hub_download(HF_REPO, f"{prefix}/{CORPUS_MANIFEST}",
                                repo_type="model", local_dir=dl_dir)
    with open(man_local) as f:
        manifest = json.load(f)
    n_parts, part_bytes = manifest["parts"], manifest["part_bytes"]

    bin_path = os.path.join(out_dir, CORPUS_TOKENS)
    open(bin_path, "ab").close()
    done_bytes = (os.path.getsize(bin_path) // part_bytes) * part_bytes
    os.truncate(bin_path, done_bytes)  # drop any torn tail
    start_part = done_bytes // part_bytes

    for i in range(start_part, n_parts):
        part = hf_hub_download(HF_REPO, f"{prefix}/tokens.bin.part{i:04d}",
                               repo_type="model", local_dir=dl_dir)
        with open(bin_path, "ab") as dst, open(part, "rb") as src:
            shutil.copyfileobj(src, dst, 64 * 1024 * 1024)
        os.remove(part)
        print(f"[corpus] assembled part {i + 1}/{n_parts}", flush=True)

    off_local = hf_hub_download(HF_REPO, f"{prefix}/{CORPUS_OFFSETS}",
                                repo_type="model", local_dir=dl_dir)
    shutil.copyfile(off_local, os.path.join(out_dir, CORPUS_OFFSETS))
    _write_manifest(out_dir, manifest)
    _load_corpus(out_dir)  # final integrity check against the manifest
    print(f"[corpus] download complete and verified: {out_dir}", flush=True)
    return out_dir
