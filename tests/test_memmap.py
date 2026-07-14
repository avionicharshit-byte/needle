import json
import os

import numpy as np
import pytest

from src.data import (
    DATASETS,
    _corpus_pools,
    _process_parquet_file,
    _write_manifest,
    memmap_block_stream,
    memmap_val_set,
)
from src.tokenizer import BOS_ID, EOS_ID


def write_corpus(tmp_path, docs):
    """Write a corpus dir from a list of token-id lists (raw, no BOS/EOS)."""
    tokens = np.concatenate([np.asarray(d, dtype=np.uint16) for d in docs])
    ends = np.cumsum([len(d) for d in docs]).astype(np.int64)
    tokens.tofile(tmp_path / "tokens.bin")
    ends.tofile(tmp_path / "offsets.bin")
    _write_manifest(str(tmp_path), {"repo": "test", "done_files": ["f0"],
                                    "docs": len(docs), "tokens": int(len(tokens))})
    return str(tmp_path)


@pytest.fixture
def corpus(tmp_path):
    # 200 docs; doc i consists of the token (100 + i) repeated, so every doc
    # is identifiable from its tokens
    docs = [[100 + i] * (5 + i % 7) for i in range(200)]
    return write_corpus(tmp_path, docs), docs


def test_val_train_disjoint(corpus):
    data_dir, docs = corpus
    val_pool, train_pool = _corpus_pools(len(docs))
    assert set(val_pool.tolist()).isdisjoint(train_pool.tolist())
    assert len(val_pool) + len(train_pool) == len(docs)

    val_tokens, val_segs = memmap_val_set(data_dir, 1, 2, 16)
    val_ids = set(val_tokens.flatten().tolist()) - {BOS_ID, EOS_ID}
    # every val token must come from a val-pool doc
    val_doc_tokens = {100 + int(i) for i in val_pool}
    assert val_ids <= val_doc_tokens

    stream = memmap_block_stream(data_dir, 2, 16, seed=1)
    train_ids = set()
    for _ in range(10):
        t, _s = next(stream)
        train_ids |= set(t.flatten().tolist())
    train_ids -= {BOS_ID, EOS_ID}
    assert train_ids.isdisjoint(val_ids)


def test_document_reconstruction(corpus):
    data_dir, docs = corpus
    stream = memmap_block_stream(data_dir, 1, 16, seed=0)
    tokens, segs = next(stream)
    # each segment within a row must be a contiguous piece of one BOS+doc+EOS
    row_t, row_s = tokens[0].tolist(), segs[0].tolist()
    for seg_id in sorted(set(row_s)):
        piece = [t for t, s in zip(row_t, row_s) if s == seg_id]
        inner = [t for t in piece if t not in (BOS_ID, EOS_ID)]
        assert len(set(inner)) <= 1  # all tokens of one doc are identical by construction


def test_determinism_and_epoch_reshuffle(corpus):
    data_dir, _ = corpus
    a = next(memmap_block_stream(data_dir, 2, 16, seed=7))
    b = next(memmap_block_stream(data_dir, 2, 16, seed=7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = next(memmap_block_stream(data_dir, 2, 16, seed=8))
    assert not np.array_equal(a[0], c[0])


def test_max_docs_fixed_subset(corpus):
    data_dir, _ = corpus
    def epoch_docs(seed):
        stream = memmap_block_stream(data_dir, 2, 16, seed=seed, max_docs=20)
        seen = set()
        for _ in range(30):  # enough to cycle the 20-doc subset several times
            t, _s = next(stream)
            seen |= set(t.flatten().tolist())
        return seen - {BOS_ID, EOS_ID}
    # same subset regardless of stream seed (subset fixed by base permutation)
    assert epoch_docs(1) == epoch_docs(99)
    assert len(epoch_docs(1)) <= 20


def test_torn_write_detected(corpus):
    data_dir, _ = corpus
    with open(os.path.join(data_dir, "tokens.bin"), "ab") as f:
        f.write(b"\x00\x00")  # one token beyond the manifest
    with pytest.raises(RuntimeError, match="manifest"):
        memmap_val_set(data_dir, 1, 2, 16)


class FakeTokenizer:
    vocab_size = 16384

    def encode(self, text):
        return [ord(c) % 250 for c in text]


def test_process_parquet_file(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "query": ["What is X?", "Explain Y.", ""],
        "synthetic_reasoning": ["think about X", "", "orphan"],
        "synthetic_answer": ["X is a thing", "Y works like so", "no query -> dropped"],
    }
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table(rows), shard)

    bin_path, off_path = tmp_path / "tokens.bin", tmp_path / "offsets.bin"
    with open(bin_path, "wb") as bf, open(off_path, "wb") as of:
        docs, tokens = _process_parquet_file(
            str(shard), DATASETS["synth"], FakeTokenizer(), bf, of, 0)

    assert docs == 2  # third row has empty query -> dropped by the formatter
    ends = np.fromfile(off_path, dtype=np.int64)
    stored = np.fromfile(bin_path, dtype=np.uint16)
    assert len(ends) == docs and ends[-1] == tokens == len(stored)
    # first doc round-trips exactly through the formatter + fake tokenizer
    text0 = DATASETS["synth"].fmt({k: v[0] for k, v in rows.items()})
    assert stored[:ends[0]].tolist() == FakeTokenizer().encode(text0)


def test_file_slice(tmp_path):
    from src.data import _FileSlice
    p = tmp_path / "blob.bin"
    data = bytes(range(256)) * 10  # 2560 bytes
    p.write_bytes(data)

    with _FileSlice(str(p), 1000, 500) as fs:
        assert fs.read() == data[1000:1500]
        fs.seek(0)
        assert fs.read(100) == data[1000:1100]
        assert fs.tell() == 100
        fs.seek(-50, 2)
        assert fs.read() == data[1450:1500]
        fs.seek(0)
        assert len(fs.read(9999)) == 500  # reads never escape the slice


def test_split_reassemble_roundtrip(tmp_path):
    """Slicing tokens.bin into parts and appending them back is lossless —
    the local core of upload_corpus/download_corpus, no network involved."""
    import shutil
    from src.data import _FileSlice

    src_path = tmp_path / "tokens.bin"
    data = np.random.default_rng(0).integers(0, 2**16, 10_000, dtype=np.uint16).tobytes()
    src_path.write_bytes(data)

    part_bytes = 4096  # deliberately not a divisor of len(data)
    n_parts = -(-len(data) // part_bytes)
    parts = []
    for i in range(n_parts):
        start = i * part_bytes
        with _FileSlice(str(src_path), start, min(part_bytes, len(data) - start)) as fs:
            part = tmp_path / f"part{i:04d}"
            part.write_bytes(fs.read())
            parts.append(part)

    out = tmp_path / "reassembled.bin"
    # simulate a resume: pretend part 0 was already assembled, with a torn tail
    out.write_bytes(parts[0].read_bytes() + b"\x00\x01\x02")
    done = (out.stat().st_size // part_bytes) * part_bytes
    os.truncate(out, done)
    for part in parts[done // part_bytes:]:
        with open(out, "ab") as dst, open(part, "rb") as sf:
            shutil.copyfileobj(sf, dst)
    assert out.read_bytes() == data
