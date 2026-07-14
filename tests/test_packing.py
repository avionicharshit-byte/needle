import numpy as np
import pytest

from src.pretraining import _pack_rows
from src.tokenizer import BOS_ID, EOS_ID


class FakeTokenizer:
    """Maps each character to ord(c) — deterministic, no artifacts needed."""

    def encode(self, text):
        return [ord(c) for c in text]


@pytest.fixture
def tok():
    return FakeTokenizer()


def collect(tok, texts, batch_size, seq_len):
    return list(_pack_rows(tok, iter(texts), batch_size, seq_len))


def test_row_shape_and_fill(tok):
    batches = collect(tok, ["abc", "de", "fgh", "ij"] * 4, batch_size=2, seq_len=7)
    assert len(batches) >= 1
    tokens, segs = batches[0]
    assert tokens.shape == (2, 8)  # seq_len + 1
    assert segs.shape == (2, 8)
    assert tokens.dtype == np.int32
    # rows are always completely filled: no padding segment ids
    assert (segs > 0).all()


def test_document_reconstruction(tok):
    """Concatenating segments across rows reproduces the [BOS]+text+[EOS] stream."""
    texts = ["hello", "world!", "abc", "xy", "packing"]
    batches = collect(tok, texts * 3, batch_size=1, seq_len=6)
    flat = np.concatenate([t[0] for t, _ in batches])
    expected = []
    for text in texts * 3:
        expected.extend([BOS_ID] + tok.encode(text) + [EOS_ID])
    # the packer drops the final partial row; compare the covered prefix
    assert flat.tolist() == expected[:len(flat)]


def test_segment_ids_restart_per_row_and_increment_per_doc(tok):
    # each doc is 4 tokens with BOS/EOS ("ab" -> B,a,b,E); row is 8 -> 2 docs/row
    batches = collect(tok, ["ab"] * 8, batch_size=1, seq_len=7)
    _, segs = batches[0]
    assert segs[0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    _, segs2 = batches[1]
    assert segs2[0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]  # counter restarts per row


def test_long_doc_spans_rows_as_fresh_segment(tok):
    # 10-token doc ("abcdefgh" + BOS/EOS) into 6-token rows
    batches = collect(tok, ["abcdefgh", "abcdefgh"], batch_size=1, seq_len=5)
    _, segs0 = batches[0]
    assert segs0[0].tolist() == [1] * 6  # first 6 tokens of doc 1
    _, segs1 = batches[1]
    # remaining 4 tokens of doc 1 restart as segment 1; doc 2 starts as segment 2
    assert segs1[0].tolist() == [1, 1, 1, 1, 2, 2]


def test_loss_mask_blocks_cross_doc_prediction(tok):
    """(seg_in == seg_tgt) & (seg_in > 0) must exclude the EOS->next-BOS step."""
    batches = collect(tok, ["ab"] * 4, batch_size=1, seq_len=7)
    tokens, segs = batches[0]
    seg_in, seg_tgt = segs[:, :-1], segs[:, 1:]
    loss_mask = (seg_in == seg_tgt) & (seg_in > 0)
    # position 3 predicts position 4 = first token of doc 2 -> masked out
    assert loss_mask[0].tolist() == [True, True, True, False, True, True, True]
    # the masked-out target is doc 2's BOS
    assert tokens[0, 4] == BOS_ID


def test_determinism(tok):
    texts = ["alpha", "beta", "gamma", "delta"] * 5
    a = collect(tok, texts, batch_size=2, seq_len=9)
    b = collect(tok, texts, batch_size=2, seq_len=9)
    for (ta, sa), (tb, sb) in zip(a, b):
        assert np.array_equal(ta, tb)
        assert np.array_equal(sa, sb)
