"""lm-evaluation-harness adapter (E6).

LoglikelihoodScorer is harness-independent (testable without lm-eval
installed); make_harness_lm wraps it in an lm_eval.api.model.LM for
lm_eval.simple_evaluate. All tasks in the E6 list (lambada, hellaswag,
arc_easy, piqa, sciq, winogrande, mmlu) score via loglikelihood only —
generate_until is intentionally unimplemented.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .architecture import make_causal_packing_mask
from .tokenizer import BOS_ID


class LoglikelihoodScorer:
    """Sum-logprob + is-greedy for (context, continuation) pairs.

    One document per row (no packing), right-padded to a fixed length so the
    jitted forward compiles once. Rows longer than seq_len+1 are truncated on
    the LEFT (context side) — the continuation always survives whole.
    """

    def __init__(self, model, params, tokenizer, seq_len=2048, batch_size=16):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._fn = None

    def _logprob_fn(self):
        if self._fn is None:
            model = self.model

            @jax.jit
            def f(params, tokens, segs):
                inputs, targets = tokens[:, :-1], tokens[:, 1:]
                mask = make_causal_packing_mask(segs[:, :-1])
                logits = model.apply({"params": params}, inputs, mask=mask)
                logprobs = jax.nn.log_softmax(logits, axis=-1)
                tgt_lp = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
                greedy = jnp.argmax(logits, axis=-1) == targets
                return tgt_lp, greedy

            self._fn = f
        return self._fn

    def encode_pair(self, context, continuation):
        """Harness-standard split: trailing context whitespace moves to the
        continuation; continuation ids come from encoding the concatenation
        so the boundary tokenizes as it would in running text."""
        n = len(context) - len(context.rstrip())
        if n:
            context, continuation = context[:-n], context[-n:] + continuation
        ctx_ids = self.tokenizer.encode(context) if context else []
        whole = self.tokenizer.encode(context + continuation)
        if whole[:len(ctx_ids)] == ctx_ids:
            cont_ids = whole[len(ctx_ids):]
        else:  # rare SP merge across the boundary — fall back to separate encode
            cont_ids = self.tokenizer.encode(continuation)
        return ctx_ids, cont_ids

    def score_token_rows(self, rows):
        """rows: list of (ids, n_cont) — score the last n_cont tokens of each.
        Returns list of (sum_logprob, is_greedy)."""
        row_len = self.seq_len + 1
        out = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            B = len(chunk)
            tokens = np.zeros((B, row_len), dtype=np.int32)
            segs = np.zeros((B, row_len), dtype=np.int32)
            spans = []
            for i, (ids, n_cont) in enumerate(chunk):
                ids = list(ids)[-row_len:]  # left-truncate, continuation is at the end
                if n_cont >= len(ids):
                    raise ValueError("continuation must leave >=1 context token (BOS)")
                tokens[i, :len(ids)] = ids
                segs[i, :len(ids)] = 1
                spans.append((len(ids) - n_cont - 1, len(ids) - 1))  # target-space
            tgt_lp, greedy = self._logprob_fn()(
                self.params, jnp.asarray(tokens), jnp.asarray(segs))
            tgt_lp, greedy = np.asarray(tgt_lp), np.asarray(greedy)
            for i, (lo, hi) in enumerate(spans):
                out.append((float(tgt_lp[i, lo:hi].sum()), bool(greedy[i, lo:hi].all())))
        return out

    def score_pairs(self, pairs):
        """pairs: list of (context, continuation) strings."""
        rows = []
        for context, continuation in pairs:
            ctx_ids, cont_ids = self.encode_pair(context, continuation)
            rows.append(([BOS_ID] + ctx_ids + cont_ids, len(cont_ids)))
        return self.score_token_rows(rows)

    def score_rolling(self, text):
        """Full-text loglikelihood in seq_len windows (BOS-anchored first window,
        1-token stride overlap after — every token scored exactly once)."""
        ids = [BOS_ID] + self.tokenizer.encode(text)
        total = 0.0
        pos = 0
        while pos < len(ids) - 1:
            window = ids[pos: pos + self.seq_len + 1]
            n_cont = min(len(window) - 1, len(ids) - 1 - pos)
            total += self.score_token_rows([(window, n_cont)])[0][0]
            pos += n_cont
        return total


def make_harness_lm(model, params, tokenizer, seq_len=2048, batch_size=16):
    from lm_eval.api.model import LM

    scorer = LoglikelihoodScorer(model, params, tokenizer, seq_len, batch_size)

    class SANLM(LM):
        def loglikelihood(self, requests):
            return scorer.score_pairs([req.args for req in requests])

        def loglikelihood_rolling(self, requests):
            return [(scorer.score_rolling(req.args[0]),) for req in requests]

        def generate_until(self, requests):
            raise NotImplementedError(
                "SAN E6 evals are loglikelihood-only; generation tasks unsupported")

    return SANLM()
