import contextlib
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .tokenizer import (
    get_tokenizer,
    BOS_ID, EOS_ID,
    IM_START_ID, IM_END_ID, THINK_START_ID, THINK_END_ID,
)
from .data import resolve_dataset, build_val_set, stream_examples, _encode_stream
from .architecture import SimpleAttentionNetwork, make_causal_packing_mask
from .run import load_checkpoint, generate

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_stdout(path):
    """Duplicate stdout into `path` — eval reports must survive pod loss."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


def upload_result(path, dest_dir="results"):
    """Push a result file to the HF repo (short retries, non-fatal)."""
    from huggingface_hub import HfApi
    from .tokenizer import HF_REPO
    for attempt in range(3):
        try:
            HfApi().upload_file(path_or_fileobj=path,
                                path_in_repo=f"{dest_dir}/{os.path.basename(path)}",
                                repo_id=HF_REPO, repo_type="model")
            print(f"[hf] results uploaded: {dest_dir}/{os.path.basename(path)}")
            return
        except Exception as e:
            print(f"[hf] results upload attempt {attempt + 1}/3 failed: {e}")
            time.sleep(10 * (attempt + 1))


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


# ---------------------------------------------------------------------------
# E4 — gap decomposition: loss sliced by token region x document group (H2)
# ---------------------------------------------------------------------------

REGIONS = ("other", "query", "trace", "answer")


def region_ids(ids):
    """Per-token region labels for one ChatML-formatted doc (ids incl. BOS/EOS).

    query(1) = user turn body; trace(2) = inside <think>; answer(3) =
    assistant text outside the trace; other(0) = BOS/EOS and the markers.
    Role-header tokens ("user\\n" / "assistant\\n") count toward their turn's
    region — 1-2 boilerplate tokens per doc, identical across arms, so Δloss
    comparisons are unaffected.
    """
    out = np.zeros(len(ids), dtype=np.int32)
    turns, region = 0, 0
    for i, t in enumerate(ids):
        if t == IM_START_ID:
            turns += 1
            region = 1 if turns == 1 else 3
        elif t == IM_END_ID:
            region = 0
        elif t == THINK_START_ID:
            region = 2
        elif t == THINK_END_ID:
            region = 3
        else:
            out[i] = region
    return out


def _pack_labeled_docs(labeled_docs, batch_size, seq_len):
    """Whole-doc packing: docs are never split across rows, so per-token label
    planes stay aligned. Rows are zero-padded (seg 0 = masked everywhere).
    Docs longer than seq_len+1 are skipped and counted.

    labeled_docs: iterable of (ids, regions, group_idx).
    Yields (tokens, segs, regions, groups) int32 arrays of shape (B, T+1);
    groups is -1 on padding. Emits the final partial batch. Returns skip count
    via the mutable `skipped` list trick is avoided — caller counts via the
    generator's .skipped attribute set on the wrapper below.
    """
    row_len = seq_len + 1
    rows = []
    cur_t, cur_s, cur_r, cur_g = [], [], [], []
    seg = 0

    def flush_row():
        nonlocal cur_t, cur_s, cur_r, cur_g, seg
        pad = row_len - len(cur_t)
        rows.append((
            np.array(cur_t + [0] * pad, dtype=np.int32),
            np.array(cur_s + [0] * pad, dtype=np.int32),
            np.array(cur_r + [0] * pad, dtype=np.int32),
            np.array(cur_g + [-1] * pad, dtype=np.int32),
        ))
        cur_t, cur_s, cur_r, cur_g = [], [], [], []
        seg = 0

    for ids, regions, group in labeled_docs:
        if len(ids) > row_len:
            continue  # caller pre-filters/counts; belt and braces
        if len(cur_t) + len(ids) > row_len:
            flush_row()
        seg += 1
        cur_t.extend(ids)
        cur_s.extend([seg] * len(ids))
        cur_r.extend(regions.tolist())
        cur_g.extend([group] * len(ids))
        if len(rows) == batch_size:
            yield tuple(np.stack(x) for x in zip(*rows))
            rows = []
    if cur_t:
        flush_row()
    while rows:
        chunk, rows = rows[:batch_size], rows[batch_size:]
        yield tuple(np.stack(x) for x in zip(*chunk))


_nll_matrix_fn_cache = {}


def _get_nll_matrix_fn(model):
    key = id(model)
    if key not in _nll_matrix_fn_cache:
        @jax.jit
        def nll_matrix(params, tokens, segs):
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            seg_in, seg_tgt = segs[:, :-1], segs[:, 1:]
            mask = make_causal_packing_mask(seg_in)
            logits = model.apply({"params": params}, inputs, mask=mask)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            valid = ((seg_in == seg_tgt) & (seg_in > 0)).astype(jnp.float32)
            return nll * valid, valid
        _nll_matrix_fn_cache[key] = nll_matrix
    return _nll_matrix_fn_cache[key]


def decompose_loss(model, params, labeled_docs, group_names, batch_size, seq_len):
    """Loss sliced by (group x region) over an iterable of
    (ids, regions, group_idx) docs. Returns {(group, region): (loss, tokens)}
    including "all" marginals on both axes.
    """
    nll_matrix = _get_nll_matrix_fn(model)
    n_groups, n_regions = len(group_names), len(REGIONS)
    nll_sum = np.zeros((n_groups + 1, n_regions + 1))
    tok_sum = np.zeros((n_groups + 1, n_regions + 1))

    for tokens, segs, regions, groups in _pack_labeled_docs(labeled_docs, batch_size, seq_len):
        nll, valid = nll_matrix(params, jnp.asarray(tokens), jnp.asarray(segs))
        nll, valid = np.asarray(nll), np.asarray(valid)
        # labels aligned to TARGET tokens (loss at position i predicts i+1)
        reg_t, grp_t = regions[:, 1:], groups[:, 1:]
        for g in range(n_groups):
            for r in range(n_regions):
                sel = (grp_t == g) & (reg_t == r) & (valid > 0)
                nll_sum[g, r] += nll[sel].sum()
                tok_sum[g, r] += sel.sum()
    nll_sum[:-1, -1] = nll_sum[:-1, :-1].sum(axis=1)
    tok_sum[:-1, -1] = tok_sum[:-1, :-1].sum(axis=1)
    nll_sum[-1, :] = nll_sum[:-1, :].sum(axis=0)
    tok_sum[-1, :] = tok_sum[:-1, :].sum(axis=0)

    out = {}
    for gi, g in enumerate(list(group_names) + ["all"]):
        for ri, r in enumerate(list(REGIONS) + ["all"]):
            if tok_sum[gi, ri] > 0:
                out[(g, r)] = (nll_sum[gi, ri] / tok_sum[gi, ri], int(tok_sum[gi, ri]))
    return out


def labeled_val_docs(tokenizer, spec, num_docs, seq_len, group_field="exercise",
                     oversample=10, seed=3407):
    """Deterministic labeled eval docs: same unshuffled-head 1-in-`oversample`
    sampling as build_val_set (seed 3407), but keeping metadata. Returns
    (docs, group_names, n_skipped) where docs = [(ids, regions, group_idx)]
    and skipped counts docs longer than seq_len+1.
    """
    import random
    rng = random.Random(seed)

    def sampled():
        for text, ex in stream_examples(spec, shuffle=False):
            if rng.random() < 1.0 / oversample:
                yield text, str(ex.get(group_field, "?") or "?")

    group_names, group_idx = [], {}
    docs, skipped = [], 0
    src = sampled()
    texts_labels = []

    def texts():
        for text, label in src:
            texts_labels.append(label)
            yield text

    for doc_ids in _encode_stream(tokenizer, texts()):
        label = texts_labels[len(docs) + skipped]
        ids = [BOS_ID] + doc_ids + [EOS_ID]
        if len(ids) > seq_len + 1:
            skipped += 1
            continue
        if label not in group_idx:
            group_idx[label] = len(group_names)
            group_names.append(label)
        docs.append((np.array(ids, dtype=np.int32), region_ids(ids), group_idx[label]))
        if len(docs) >= num_docs:
            break
    return docs, group_names, skipped


def print_decomposition(matrix, group_names):
    cols = list(REGIONS) + ["all"]
    print(f"\n{'group':<24}" + "".join(f"{c:>12}" for c in cols) + f"{'tokens':>12}")
    for g in list(group_names) + ["all"]:
        cells = [matrix.get((g, c)) for c in cols]
        row = "".join(f"{c[0]:>12.4f}" if c else f"{'-':>12}" for c in cells)
        total = matrix.get((g, "all"))
        print(f"{g:<24}{row}{total[1] if total else 0:>12,}")

    grand = matrix[("all", "all")][0] * matrix[("all", "all")][1]
    print(f"\n{'loss share %':<24}" + "".join(f"{c:>12}" for c in cols))
    for g in list(group_names) + ["all"]:
        row = ""
        for c in cols:
            cell = matrix.get((g, c))
            row += f"{100 * cell[0] * cell[1] / grand:>12.2f}" if cell else f"{'-':>12}"
        print(f"{g:<24}{row}")


# ---------------------------------------------------------------------------
# E6 — downstream evals via lm-evaluation-harness
# ---------------------------------------------------------------------------

class LoglikelihoodScorer:
    """Sum-logprob + is-greedy for (context, continuation) pairs — the
    harness-independent core of the E6 adapter (testable without lm-eval).

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
        """Full-text loglikelihood in seq_len windows, every token scored once."""
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
    """Wrap a LoglikelihoodScorer in an lm_eval.api.model.LM. All E6 tasks
    score via loglikelihood only — generate_until is intentionally absent."""
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


def evaluate_downstream(model, params, tokenizer, tasks, seq_len=2048, batch_size=16):
    """E6: 0-shot loglikelihood tasks via lm-evaluation-harness."""
    if not tasks:
        return {}
    try:
        import lm_eval
    except ImportError:
        raise SystemExit("E6 downstream evals need the harness: pip install lm-eval")

    lm = make_harness_lm(model, params, tokenizer, seq_len=seq_len, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=list(tasks))["results"]
    print(f"\n{'task':<20}{'metric':<16}{'value':>10}")
    for task, metrics in results.items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{task:<20}{metric:<16}{value:>10.4f}")
    return results


def main(args):
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if getattr(args, "tasks", None):
        mode = "e6"
    elif getattr(args, "by_exercise", False) or getattr(args, "by_region", False):
        mode = "e4"
    else:
        mode = "val"
    out_path = os.path.join("results", f"{stem}__{mode}.txt")
    with tee_stdout(out_path):
        _main(args)
    if getattr(args, "upload_results", False):
        upload_result(out_path)


def _main(args):
    params, config = load_checkpoint(args.checkpoint)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    print(f"Model: {config.num_layers}L d={config.d_model} "
          f"{'no-FFN (SAN)' if config.no_feedforward else f'FFN d_ff={config.d_ff}'}")

    spec = resolve_dataset(args.dataset, getattr(args, "text_field", None))
    seq_len = min(args.seq_len, config.max_seq_len)

    tasks = getattr(args, "tasks", None)
    if tasks:
        evaluate_downstream(model, params, tokenizer, tasks,
                            seq_len=seq_len, batch_size=args.batch_size)
        return

    by_exercise = getattr(args, "by_exercise", False)
    by_region = getattr(args, "by_region", False)
    if by_exercise or by_region:
        print(f"Gap decomposition over {args.val_docs} val docs "
              f"(group field: {args.group_field})...")
        docs, group_names, skipped = labeled_val_docs(
            tokenizer, spec, args.val_docs, seq_len, group_field=args.group_field)
        if not by_exercise:
            docs = [(ids, regions, 0) for ids, regions, _ in docs]
            group_names = ["all-docs"]
        matrix = decompose_loss(model, params, docs, group_names,
                                args.batch_size, seq_len)
        if skipped:
            print(f"({skipped} docs skipped: longer than seq_len+1)")
        print_decomposition(matrix, group_names)
        return

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
