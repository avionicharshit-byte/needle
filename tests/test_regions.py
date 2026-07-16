import jax
import numpy as np
import pytest

from src.architecture import SimpleAttentionNetwork, TransformerConfig
from src.data import _fmt_synth
from src.eval import REGIONS, region_ids, _pack_labeled_docs, decompose_loss
from src.tokenizer import (
    BOS_ID, EOS_ID,
    IM_START_ID, IM_END_ID, THINK_START_ID, THINK_END_ID,
)


def test_marker_ids_match_tokenizer():
    """The hardcoded 4-7 ids must match the trained tokenizer's pieces, and
    markers must stay atomic inside running text (standalone encoding picks
    up the dummy-prefix '▁', so test in context)."""
    from src.tokenizer import get_tokenizer, IM_START, IM_END, THINK_START, THINK_END
    tok = get_tokenizer()
    for marker, mid in [(IM_START, IM_START_ID), (IM_END, IM_END_ID),
                        (THINK_START, THINK_START_ID), (THINK_END, THINK_END_ID)]:
        assert tok.sp.id_to_piece(mid) == marker
        ids = tok.encode(f"x{marker}y")
        assert ids.count(mid) == 1


def test_region_ids_state_machine():
    # BOS <im_start> q q <im_end> <im_start> hdr <think> t t </think> a <im_end> EOS
    ids = [BOS_ID, IM_START_ID, 30, 31, IM_END_ID,
           IM_START_ID, 40, THINK_START_ID, 50, 51, THINK_END_ID, 60, IM_END_ID, EOS_ID]
    expected = [0, 0, 1, 1, 0,
                0, 3, 0, 2, 2, 0, 3, 0, 0]
    assert region_ids(ids).tolist() == expected


def test_region_ids_real_doc():
    """Formatted SYNTH doc through the real tokenizer: every region present,
    trace strictly between the think markers."""
    from src.tokenizer import get_tokenizer
    tok = get_tokenizer()
    text = _fmt_synth({"query": "What is water?",
                       "synthetic_reasoning": "H2O is a molecule.",
                       "synthetic_answer": "Water is H2O."})
    ids = [BOS_ID] + tok.encode(text) + [EOS_ID]
    regions = region_ids(ids)
    assert set(np.unique(regions)) == {0, 1, 2, 3}
    think_open = ids.index(THINK_START_ID)
    think_close = ids.index(THINK_END_ID)
    assert (regions[think_open + 1:think_close] == 2).all()
    assert (regions[think_close + 1:-2] == 3).all()  # answer up to final im_end


def test_pack_labeled_docs_alignment():
    def doc(n, group, fill):
        ids = np.full(n, fill, dtype=np.int32)
        return ids, np.full(n, group % 4, dtype=np.int32), group

    docs = [doc(5, 0, 100), doc(4, 1, 200), doc(6, 2, 300)]
    blocks = list(_pack_labeled_docs(docs, batch_size=4, seq_len=7))
    tokens = np.concatenate([b[0] for b in blocks], axis=0)
    segs = np.concatenate([b[1] for b in blocks], axis=0)
    groups = np.concatenate([b[3] for b in blocks], axis=0)

    # docs are never split: each (row, seg) pair holds exactly one whole doc
    seen = {}
    for row in range(tokens.shape[0]):
        for seg in set(segs[row].tolist()) - {0}:
            sel = segs[row] == seg
            fills = set(tokens[row][sel].tolist())
            assert len(fills) == 1  # one doc's fill value only
            seen[fills.pop()] = (int(sel.sum()), int(groups[row][sel][0]))
    assert seen == {100: (5, 0), 200: (4, 1), 300: (6, 2)}
    # padding is seg 0 / group -1
    assert (groups[segs == 0] == -1).all()


def test_decompose_loss_marginals():
    cfg = TransformerConfig(vocab_size=64, d_model=32, num_heads=4, num_kv_heads=2,
                            num_layers=2, max_seq_len=32, dtype="float32")
    model = SimpleAttentionNetwork(cfg)
    params = model.init({"params": jax.random.PRNGKey(0)},
                        np.ones((1, 8), dtype=np.int32))["params"]

    rng = np.random.default_rng(0)
    docs = []
    for g in (0, 0, 1):
        ids = rng.integers(8, 60, size=10).astype(np.int32)
        regions = np.tile([1, 2], 5).astype(np.int32)
        docs.append((ids, regions, g))
    matrix = decompose_loss(model, params, docs, ["ex_a", "ex_b"],
                            batch_size=2, seq_len=16)

    # marginals are token-weighted means of the cells
    all_loss, all_toks = matrix[("all", "all")]
    assert all_toks == 3 * 9  # 10-token docs -> 9 targets each
    acc = sum(matrix[(g, "all")][0] * matrix[(g, "all")][1] for g in ("ex_a", "ex_b"))
    assert np.isclose(acc / all_toks, all_loss)
    assert ("ex_a", "query") in matrix and ("ex_b", "trace") in matrix
