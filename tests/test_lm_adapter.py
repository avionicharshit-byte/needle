import jax
import jax.numpy as jnp
import numpy as np

from src.architecture import SimpleAttentionNetwork, TransformerConfig, make_causal_mask
from src.eval import LoglikelihoodScorer
from src.spectra import spectra
from src.tokenizer import BOS_ID


class FakeTokenizer:
    """Whitespace word tokenizer with a stable id per word (prefix property
    holds, so encode_pair takes the fast path)."""
    vocab_size = 64

    def encode(self, text):
        return [10 + (hash(w) % 50) for w in text.split()]


def tiny_scorer(seq_len=24, batch_size=2):
    cfg = TransformerConfig(vocab_size=64, d_model=32, num_heads=4, num_kv_heads=2,
                            num_layers=2, max_seq_len=seq_len, dtype="float32")
    model = SimpleAttentionNetwork(cfg)
    params = model.init({"params": jax.random.PRNGKey(0)},
                        np.ones((1, 8), dtype=np.int32))["params"]
    return LoglikelihoodScorer(model, params, FakeTokenizer(),
                               seq_len=seq_len, batch_size=batch_size), model, params


def manual_logprobs(model, params, ids):
    """Reference: plain causal forward on the unpadded row."""
    tokens = jnp.asarray([ids], dtype=jnp.int32)
    logits = model.apply({"params": params}, tokens[:, :-1],
                         mask=make_causal_mask(len(ids) - 1))
    lp = jax.nn.log_softmax(logits, axis=-1)
    tgt = np.asarray(tokens[0, 1:])
    return np.asarray(lp[0])[np.arange(len(ids) - 1), tgt]


def test_score_matches_manual_forward():
    scorer, model, params = tiny_scorer()
    ids = [BOS_ID, 12, 33, 47, 19, 25]
    n_cont = 3
    (got, _), = scorer.score_token_rows([(ids, n_cont)])
    want = manual_logprobs(model, params, ids)[-n_cont:].sum()
    assert np.isclose(got, want, rtol=1e-4), (got, want)


def test_is_greedy_flag():
    scorer, model, params = tiny_scorer()
    ctx = [BOS_ID, 12, 33]
    # teacher-forced argmax continuation must score is_greedy=True
    ids = list(ctx)
    for _ in range(3):
        logits = model.apply({"params": params},
                             jnp.asarray([ids], dtype=jnp.int32),
                             mask=make_causal_mask(len(ids)))
        ids.append(int(np.asarray(logits)[0, -1].argmax()))
    (_, greedy), = scorer.score_token_rows([(ids, 3)])
    assert greedy
    # corrupt one continuation token -> not greedy anymore (token 0 is argmin-safe)
    worst = int(np.asarray(model.apply({"params": params},
                jnp.asarray([ids[:-1]], dtype=jnp.int32),
                mask=make_causal_mask(len(ids) - 1)))[0, -1].argmin())
    (_, greedy2), = scorer.score_token_rows([(ids[:-1] + [worst], 3)])
    assert not greedy2


def test_left_truncation_keeps_continuation():
    scorer, model, params = tiny_scorer(seq_len=8)
    long_ids = [BOS_ID] + list(range(10, 30))  # 21 tokens > seq_len+1 = 9
    (got, _), = scorer.score_token_rows([(long_ids, 4)])
    want = manual_logprobs(model, params, long_ids[-9:])[-4:].sum()
    assert np.isclose(got, want, rtol=1e-4)


def test_encode_pair_moves_trailing_whitespace():
    scorer, _, _ = tiny_scorer()
    ctx_ids, cont_ids = scorer.encode_pair("the cat ", "sat down")
    whole = FakeTokenizer().encode("the cat sat down")
    assert ctx_ids + cont_ids == whole
    assert len(cont_ids) == 2  # "sat down"


def test_batching_matches_single():
    scorer, _, _ = tiny_scorer(batch_size=2)
    rows = [([BOS_ID, 11, 22, 33], 2), ([BOS_ID, 44, 55], 1),
            ([BOS_ID, 13, 14, 15, 16], 3)]
    batched = scorer.score_token_rows(rows)
    singles = [scorer.score_token_rows([r])[0] for r in rows]
    for (a, ga), (b, gb) in zip(batched, singles):
        assert np.isclose(a, b, rtol=1e-4) and ga == gb


def test_sv_spectra_shapes():
    rng = np.random.default_rng(0)
    params = {"stack": {"block": {"q": {"kernel": rng.normal(size=(3, 16, 16))},
                                  "scale": np.ones(16)}},
              "emb": {"embedding": rng.normal(size=(64, 16))}}
    out = spectra(params)
    assert list(out.keys()) == ["stack/block/q"]
    assert out["stack/block/q"]["sv"].shape == (3, 16)
    assert (out["stack/block/q"]["eff_rank"] <= 16).all()
    assert (out["stack/block/q"]["stable_rank"] >= 1).all()
