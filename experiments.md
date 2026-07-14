# Simple Attention Networks — NeurIPS Main-Track Experiment Plan

Target: NeurIPS 2027 main track (abstract deadline ~May 2027).
Status: draft v1, 2026-07-14.

---

## 1. The claim, stated so it can be falsified

The current doc frames SAN as a system (no FFN + encoder-decoder + gated residuals +
ZCRMSNorm + Muon + QAT + loss weighting + contrastive head). A systems paper with 8
entangled techniques and one task will not survive main-track review. The paper should
make one scientific claim, with the components as supporting recipe:

> **H1 (dispensability).** On tasks whose requisite knowledge is fully present in the
> input context (retrieval-and-assembly tasks: function calling, extraction, copying),
> attention-only transformers match FFN transformers at equal parameter count — and beat
> them at equal inference cost.
>
> **H2 (boundary).** The gap reverses on tasks requiring parametric knowledge
> (closed-book QA, fact completion): FFN layers earn their parameters exactly when
> knowledge must live in weights. This operationalizes the "FFNs are key-value
> memories" hypothesis (Geva et al., 2021) as an architecture-selection rule.
>
> **H3 (trainability).** Deep attention-only stacks are trainable only with the right
> stabilizers. Pure attention suffers rank collapse (Dong et al., 2021); residuals
> alone slow but don't stop it without FFN's per-position rewriting. Gated residuals +
> zero-centered norms + orthogonalized updates (Muon) jointly prevent collapse — we
> show this both in training curves and in measured representation rank across depth.

H2 is what elevates this from "we made a small model for tool calling" to a
generalizable finding: **FFNs store, attention routes; externalize the knowledge and
you can delete the store.** H3 gives the mechanistic story reviewers want.

### Positioning against prior work (reviewers will check)

| Prior work | Relationship | What we must do |
|---|---|---|
| Dong et al. 2021, "Attention is not all you need" (rank collapse) | Proves pure attention collapses; says MLPs+residuals prevent it | Our H3 is the empirical counterpoint: MLPs are not necessary for collapse-prevention if updates are orthogonalized and residuals gated. Measure rank directly. |
| Geva et al. 2021, FFN = key-value memories | Motivates H2 | Cite as hypothesis source; our contribution is the causal/architectural test |
| ReZero, SkipInit, LayerScale, Highway nets | Gated residual is in this family | Ablate against ReZero (α·F(x), α=0 init) and LayerScale explicitly; claim recipe, not novelty |
| nGPT, DeepSeek-V3 zero-centered norms | ZCRMSNorm source (already cited) | Ablate; claim recipe, not novelty |
| Muon (Jordan et al. 2024; Moonlight) | Optimizer we repurpose | Novel angle: Muon as *enabler of FFN-free depth*, not just faster training |
| gMLP / "Pay Attention to MLPs" | The mirror-image ablation (drop attention, keep MLP) | Cite as symmetric evidence that either primitive can carry tasks matched to its inductive bias |
| TinyAgent, Octopus v2, xLAM, FunctionGemma, ToolACE/APIGen | Small function-calling models | These are baselines, not competitors for the scientific claim |

---

## 2. Experiment matrix

### E1 — Main comparison: author-trained controls (the load-bearing experiment)

Comparing a bespoke 26M model against off-the-shelf 270M–600M generalists is not a
controlled comparison and reviewers will say so. Every control below is trained by us
with **identical tokenizer (8192 BPE), data, data order, steps, and eval**; only the
architecture axis varies.

| ID | Architecture | Params | Purpose |
|---|---|---|---|
| C0 | SAN (12enc/8dec, d=512, no FFN) | 26M | ours |
| C1 | Enc-dec **with FFN**, depth/width reduced to match params (e.g. d=384, 8enc/6dec, d_ff=4d) | 26M | iso-parameter |
| C2 | Enc-dec with FFN, same 12enc/8dec, d=512, d_ff=2048 | ~57M | iso-depth (shows param efficiency) |
| C3 | Enc-dec with FFN, iso-*training-FLOPs* to C0 | free | iso-compute |
| C4 | Decoder-only, no FFN, iso-param | 26M | isolates enc-dec choice |
| C5 | Decoder-only with FFN, iso-param (a "normal tiny LM") | 26M | the strongest conventional control |
| C6 | Flan-T5-small finetuned on our post-train data | 77M | public enc-dec reference point |

The codebase already supports C1/C2 via `no_feedforward=False` and `d_ff` in
`TransformerConfig` (architecture.py); C4/C5 need a decoder-only path (~small change:
encoder-less config with tools+query in the decoder context).

**Primary metrics:** exact match, call F1, name F1, JSON parse rate (existing
`benchmark_tool_calls`), on internal test set + public benchmarks (E8).
**Headline figure:** accuracy vs. params and accuracy vs. inference latency (Pareto),
with C0 dominating at iso-param and iso-latency.

### E2 — Component ablations (one-factor-at-a-time from C0)

Run at the ablation tier (§5): ~20B pretrain tokens + full post-train, 3 seeds each.

| Axis | Variants | Prediction |
|---|---|---|
| Residual | gated σ(g), g=0 (ours) · standard `x+F(x)` · ReZero `x+αF(x)`, α=0 · LayerScale · no residual | no-residual fails to train ≥12 layers; standard trains but worse; gated ≈ ReZero or slightly better |
| Norm | ZCRMSNorm · RMSNorm(γ=1) · no QK-norm | ZCRMSNorm helps stability at high Muon LR |
| Optimizer | Muon+AdamW (ours) · AdamW only · Muon on everything | AdamW-only degrades/diverges as depth grows **only when FFN is absent** — the interaction Muon×no-FFN is H3's key cell. Run the 2×2: {Muon, AdamW} × {FFN, no-FFN}. |
| INT4 QAT | on (every 100 steps) · off · every step · PTQ-only | QAT-on ≥ QAT-off in bf16 eval (regularization claim) AND >> PTQ in INT4 eval (deployment claim). If bf16 claim fails, drop the regularization framing. |
| Loss weighting | 1/2/1.5/4 (ours) · uniform · structure-only | weighted wins on value accuracy at equal exact-match elsewhere |
| Contrastive head | joint @0.1× (ours) · off · separate retriever, same size | measure both generation quality (does aux loss hurt?) and retrieval R@k |
| Depth allocation | 12/8 (ours) · 8/12 · 16/4 · 10/10 at fixed total | encoder-heavy wins (tools need bidirectional encoding) |

Report a single ablation table: Δ exact-match and Δ pretrain loss per removed
component, mean ± std over 3 seeds.

### E3 — Scaling: does the FFN-free advantage survive scale?

The most likely reviewer objection: "this only works at 26M." Answer it directly.

- **Parameter scaling:** {5M, 13M, 26M, 60M, 125M} × {SAN, iso-param FFN control},
  compute-matched (Chinchilla-style token budgets per size). Plot task metric and
  pretrain loss vs. params; locate the crossover if one exists. *An honest crossover
  strengthens H2 — FFNs win when scale lets them memorize.*
- **Data scaling:** C0 at {10B, 30B, 100B, 200B} tokens.
- **Depth scaling (H3):** attention-only at {8, 20, 32, 48} total layers, fixed params
  (shrink d), with/without gates and Muon → trainability frontier.

### E4 — Boundary conditions: where FFN-free *should* lose (H2)

Same C0 vs C1 pair, finetuned per task:

- **In-context tasks (predict SAN ≥ FFN):** function calling (ours), extractive QA
  (SQuAD-style), slot filling (SNIPS/MTOP), RAG-style QA where evidence is in the
  encoder, synthetic copy/lookup tasks.
- **Parametric tasks (predict SAN < FFN):** closed-book QA (TriviaQA subset, no
  context), LAMA-style fact completion, 2–3 digit arithmetic.
- **Dose-response:** RAG-QA where the gold passage is included with probability p ∈
  {0, 0.5, 1}. Prediction: the SAN–FFN gap shrinks monotonically as p→1. This is the
  single most persuasive figure for H2.

### E5 — Mechanistic analysis (H3)

- **Rank collapse:** effective rank / entropy of token representations per layer, for
  C0 vs. ablations (no gates, AdamW-only, with FFN). Directly tests Dong et al.'s
  prediction in our setting.
- **Gate trajectories:** learned σ(g) per layer over training — which layers sharpen
  (→1) vs. self-prune (→0). Connects to depth-allocation ablation.
- **Singular value spectra** of Q/K/V/O with vs. without Muon.
- **Copy-head evidence:** cross-attention alignment between generated argument-value
  tokens and their source spans in the query (precision of argmax attention on the
  copy source). Quantifies "tool calling is retrieval-and-assembly."

### E6 — Tool-set scaling and the contrastive head

- Accuracy vs. number of tools in context: {1, 2, 4, 8, 16, 32, 64} tools, with
  distractors sampled adversarially (similar names/descriptions).
- With vs. without contrastive top-k filtering at large tool counts; retrieval R@k
  (existing `benchmark_retrieval`); end-to-end accuracy of retrieve-then-generate vs.
  all-tools-in-context, and the latency saved.
- Irrelevance detection: no-applicable-tool cases (maps to BFCL relevance category).

### E7 — Efficiency on real edge hardware

Claims of 6000/1200 tok/s must become controlled measurements:

- Devices: one iPhone (A17+), one Android mid-tier, one M-series Mac — via Cactus.
- Metrics: prefill tok/s, decode tok/s, time-to-first-token, peak RSS, energy/query
  (where measurable), INT4 model size.
- Baselines at INT4: FunctionGemma-270m, Qwen3-0.6B, LFM2-350m, SmolLM2-360m on
  llama.cpp/ExecuTorch — same prompts, same device, median of ≥50 runs.
- KV-cache memory vs. context length: enc-dec fixed cross-KV vs. decoder-only growth.
- Headline: **accuracy-per-millisecond Pareto frontier** (ties E1 to E7).

### E8 — Public benchmarks and external baselines

- **BFCL v3/v4:** single-turn AST categories (simple, multiple, parallel) +
  relevance. Near-mandatory for credibility in this subfield.
- **API-Bank** L1; one of ToolBench/Seal-Tools as a third external set.
- Baselines evaluated two ways: (a) zero-shot as released, (b) **finetuned on our
  exact post-train data** — (b) is the fair comparison and the one reviewers trust.
- Public-data reproducibility run: C0 post-trained only on public data
  (xLAM-function-calling-60k / Glaive-v2 / ToolACE) so the result doesn't depend on
  our private Gemini-generated set. Release weights for this variant.

---

## 3. Statistical methodology

- 3 seeds per trained config at ablation tier; 5 seeds for headline C0/C1/C5.
  Mean ± 95% CI everywhere; paired bootstrap (10k resamples) for C0-vs-control deltas
  on shared test items; report p-values only for headline claims.
- Fixed data order across configs within a comparison (same shuffle seed).
- Decoding fixed at greedy + constrained JSON (`model/constrained.py`) for all systems
  that support it; report unconstrained numbers in appendix.
- All hyperparameters in appendix; HP budget disclosed: controls C1–C5 get an equal
  LR sweep (≥5 points) — "we tuned ours and not the baseline" is a desk-reject risk.
- Release: code, tokenizer, weights, data-gen pipeline, public-data variant, eval
  harness commit hash.

## 4. What must be built (gap analysis vs. current repo)

| Need | Status |
|---|---|
| FFN on/off, d_ff, depth configs | done (`TransformerConfig`) |
| Decoder-only control (C4/C5) | ~2–3 days work |
| ReZero/LayerScale residual variants | trivial flag in blocks |
| AdamW-only / Muon-everywhere | trivial (`optim.py` labels) |
| Rank/entropy + gate logging | small instrumentation hooks |
| BFCL/API-Bank adapters | ~1 week (format conversion + scoring) |
| Boundary-condition task loaders (SQuAD, TriviaQA, SNIPS, synthetic) | ~1–2 weeks |
| On-device benchmark scripts (Cactus + llama.cpp) | ~1 week |
| Multi-seed launcher + results DB | ~2–3 days |

## 5. Compute budget (16× TPU v6e reference: 200B-token pretrain ≈ 27 h ≈ 432 chip-h)

| Tier | Token budget | Cost/run | Runs | Total (chip-h) |
|---|---|---|---|---|
| Ablation (E2, E4-small, E3-depth) | 20B | ~43 | ~30 cfg × 3 seeds = 90 | ~3,900 |
| Scaling (E3) | Chinchilla per size × 2 arch | varies | ~10 | ~2,500 |
| Headline (C0, C1, C5 @ 200B × 5 seeds) | 200B | ~432 (26M) | 15 | ~6,500 |
| Post-train/finetune/eval/on-device | 2B each | small | many | ~600 |
| **Total** | | | | **~13,500 chip-h ≈ 35 days on the 16-chip pod** |

If that's too much: drop headline seeds to 3, cut scaling to 4 sizes, run E2 at 10B
tokens → ~8,000 chip-h.

## 6. Risks and fallback framings

- **C1 (iso-param FFN) beats C0 on tool calling** → the dispensability claim dies as
  stated; reframe on the Pareto axis (E7): equal accuracy at 2–3× lower latency/memory.
  Still publishable, weaker venue fit.
- **No boundary effect in E4** → H2 dies; paper becomes an efficiency+recipe paper;
  consider MLSys/EMNLP instead of NeurIPS.
- **Gated residual ≈ ReZero exactly** → fine; fold into recipe, don't claim novelty.
- **QAT regularization claim fails in bf16 eval** → keep QAT for deployment only.
- **Baselines finetuned on our data close the gap** → lean on iso-cost story + E6/E7.

## 7. Priority order

1. **E1 controls (C1, C5) at ablation tier** — if C0 doesn't hold up here, everything
   else is moot. Do this first, before any infra polish.
2. E4 dose-response (the H2 figure) at small scale.
3. E2 ablation grid + E5 rank measurements (H3).
4. E3 scaling curves.
5. E8 public benchmarks + baseline finetuning; E7 on-device; headline-tier reruns.
6. Writing: intro around H1/H2/H3, not around the Needle product.
