# Simple Attention Networks — Experiment Design

Target: NeurIPS main track. Status: draft v2, 2026-07-14 (supersedes the
tool-calling-era plan; this branch studies SAN on general pretraining).

Hardware: one 8×H100 node (RunPod). Framework: JAX pmap, flash attention,
bf16 compute. Data: PleIAs/SYNTH (~80M docs ≈ 70B tokens under our tokenizer),
single-turn ChatML with reasoning traces, 16k BPE, seq_len 2048, doc-boundary
masked packing.

---

## 1. Hypotheses

> **H1 — dispensability.** On reasoning-oriented pretraining (SYNTH), a
> decoder-only attention-only transformer matches a standard transformer at
> equal parameter count on val loss and downstream evals. Reasoning traces
> externalize intermediate computation into context — the regime where
> attention (routing/copying over visible tokens) should carry the load and
> per-position FFN transformation should matter least.
>
> **H2 — gap decomposition.** Whatever SAN–FFN gap exists concentrates on
> parametric-knowledge demands. Measured two ways SYNTH makes cheap:
> (a) by exercise type (memorization vs math/rag/writing — `exercise` field),
> (b) by token region (trace vs answer — atomic `<think>`/`</think>` marker
> ids). Prediction: gap largest on memorization-answers, smallest on traces.
> Operationalizes "FFNs store, attention routes" (Geva et al. 2021).
>
> **H3 — trainability.** Deep attention-only stacks train only with the right
> stabilizers. Pure attention rank-collapses (Dong et al. 2021); we claim
> gated residuals + ZCRMSNorm + Muon prevent it without FFN, and show it in
> training curves AND measured per-layer effective rank (`--log-rank-every`).

Positioning: same as before — vs Dong et al. (counterpoint), Geva et al.
(hypothesis source), ReZero/LayerScale (gated residual family — recipe, not
novelty), nGPT/DeepSeek-V3 (ZCRMSNorm source), Muon (novel angle: enabler of
FFN-free depth), gMLP (mirror-image evidence). New: PleIAs Monad-56M /
Baguettotron-321M — same data, same format, published numbers = free external
reference points for our controls.

## 2. Fixed experimental frame (identical across ALL arms)

- Tokenizer (16,384 BPE, trained on ChatML-formatted SYNTH), seq_len 2048,
  packing, loss masking, z-loss, WSD schedule shape, warmup/decay ratios,
  data order (same `--seed` within a comparison), val set (seeded window
  subsample, disjoint from train), eval cadence.
- LR conventions: Adam `--lr × 8`, Muon `--muon-lr × √8`. Weight decay 0.01
  on Dense kernels in both optimizer arms.
- **Params reported as non-embedding first** — tied 16k×d embeddings are
  identical within every pair and would otherwise flatter small models.
- Default token budget: 100k steps × 512 rows × 2048 = **105B tokens**
  (~1.5 passes over SYNTH; PleIAs report convergence at 100–200B).

## 3. Configurations (measured with `jax.eval_shape`, vocab 16,384)

| ID | Config | Total params | Non-emb | Train GFLOPs/tok* | Role |
|---|---|---|---|---|---|
| SAN | 20L d512, no FFN | 24.13M | 15.74M | ~0.40 | ours |
| FFN-isoP | 4L d512 ff2048 | 24.12M | 15.73M | ~0.20 | iso-parameter (Δ < 0.04%) |
| FFN-isoF | 9L d512 ff2048 | ~43M | ~35M | ~0.39 | iso-training-FLOPs |
| FFN-isoD | 20L d512 ff2048 | 87.06M | 78.67M | ~0.72 | iso-depth (param-efficiency angle) |

*at T=2048 incl. attention quadratic term; halve for causal effective. The
FLOP asymmetry is the finding's honest frame: at iso-param SAN spends ~2× the
FLOPs/token; at iso-FLOP the FFN arm gets ~2.2× the params. Report all three
matchings — a reviewer can pick their preferred axis and we win or lose on it
explicitly.

Scaling ladder (SAN side; FFN-isoP partners computed the same way):

| Size | Config | Total | Non-emb |
|---|---|---|---|
| tiny | 10L d256 | 6.16M | 1.97M |
| small | 14L d384 | 12.49M | 6.20M |
| base | 20L d512 | 24.13M | 15.74M |
| large | 26L d640 | 42.46M | 31.97M |
| xl | 32L d768 | 69.24M | 56.65M |

## 4. Experiment matrix

### E0 — LR fairness sweep (before anything headline)
Per arm {SAN, FFN-isoP} × {muon, adamw}: 3-point LR sweep (0.5×, 1×, 2× of
defaults) at 5B tokens. Best LR per arm carries forward. "We tuned ours and
not the control" is a desk-reject risk; this is the insurance.
~12 runs × 5B = 60B tokens.

### E1 — Headline comparison
SAN vs FFN-isoP vs FFN-isoF vs FFN-isoD at 105B tokens, muon, seed 42; then
+2 seeds for SAN and FFN-isoP (the load-bearing pair).
6 × 105B + 2 × 105B = ~840B tokens.
Primary metric: val loss/PPL; secondary: downstream evals (E6), throughput.

### E2 — The optimizer×architecture 2×2 (H3 core)
{muon, adamw} × {SAN, FFN-isoP} at 30B tokens. The prediction that matters:
adamw hurts the SAN arm *more* than the FFN arm (interaction term, not main
effect). Already flag-complete: `--optimizer`, `--ffn`.
2 remaining cells × 30B = 60B tokens.

### E3 — Component ablations (20B tokens each, SAN arm)
| Axis | Variants | Needs code |
|---|---|---|
| Residual | gated (ours) · ReZero (α init 0, no sigmoid) · standard `x+F(x)` · none | flag |
| Norm | ZCRMSNorm (ours) · RMSNorm γ=1 · no QK-norm | flag |
| Post-attn norm | off (ours) · sandwich `x+σ(g)·Norm(Attn(Norm(x)))` | flag |
| Depth at iso-param | {8, 20, 32, 48}L (d adjusted) × {gates on/off} | configs only |
~14 runs × 20B = 280B tokens. Report one table: Δval-loss per component,
mean±std over 3 seeds for any variant within noise of the default.

### E4 — Gap decomposition (H2; evaluation of E1/E2 checkpoints — no new training)
- Per-exercise val loss: separate packed val sets filtered by `exercise`
  (memorization / mcq / math / rag / creative-writing).
- Per-region val loss: split token positions by `<think>`…`</think>` marker
  ids (4–7) into query / trace / answer regions.
- The H2 figure: SAN−FFN Δloss per (exercise × region) cell, with the
  memorization-answer cell predicted worst and trace cells predicted best.

### E5 — Data scaling / repetition (uses `--max-docs`)
SAN + FFN-isoP at 30B tokens with `--max-docs` ∈ {2M, 8M, 32M, full}
(≈17×, 4×, 1×, 0.4× repetition). Secondary H2 probe: FFN arms should benefit
more from repetition (memorization capacity), SAN should degrade less under
data constraint. 8 runs × 30B = 240B tokens.

### E6 — Downstream evals (no training; needs lm-eval adapter)
0-shot loglikelihood tasks suitable for <100M models: lambada, hellaswag,
arc_easy, piqa, sciq, winogrande, mmlu. Direct comparison against published
Monad-56M and Baguettotron-321M numbers (same data + format). Run on every
E1 checkpoint.

### E7 — Mechanistic (H3; instrumentation mostly built)
- Rank trajectories: `--log-rank-every 2500` on all E1/E2/E3 runs (free).
- Gate trajectories: per-layer σ(g) over training (already logged).
- Singular-value spectra of Q/K/V/O kernels: muon vs adamw arms, at 0/25/50/
  100% of training (offline script over checkpoints).
- Money figure: per-layer effective rank at end of training for {SAN+muon,
  SAN+adamw, SAN-no-gates, FFN-isoP} — Dong et al.'s prediction tested.

## 5. Figures the paper needs (working list)

1. Val loss vs tokens: SAN vs three FFN matchings (E1).
2. The H2 heatmap: Δloss by exercise × region (E4).
3. Scaling: val loss vs non-emb params, both architectures, crossover marked
   if present (E1 ladder subset — run {tiny, small, base, large} × 2 at 30B).
4. Rank-vs-depth panel (E7).
5. Gate trajectories heatmap (layers × training time).
6. Optimizer×architecture interaction bars (E2).
7. Downstream table vs Monad/Baguettotron (E6).

## 6. Infrastructure to build (gap list)

| Need | Size |
|---|---|
| Residual/norm variant flags (ReZero, standard, RMSNorm, no-QK, sandwich) | small, architecture.py |
| Per-exercise val sets (filter by `exercise` before packing) | small, data.py + eval.py |
| Per-region loss slicing (marker ids 4–7 → region masks) | small, eval.py |
| lm-eval loglikelihood adapter (stub exists in eval.py) | ~2 days |
| SV-spectra script over checkpoints | small |
| Multi-run launcher (shell loop over configs × seeds is fine) | trivial |
| Param/FLOP calculator (one-liner exists; commit as script) | trivial |

## 7. Statistical methodology

- 3 seeds for the headline pair; 1 seed + noise-band from headline for
  ablation cells (upgrade any surprising cell to 3 seeds before believing it).
- Same data seed across arms within a comparison; val set fixed (seed 3407).
- Mean ± 95% CI on val loss; paired-by-block bootstrap for Δloss claims.
- Every hyperparameter in the appendix; E0 sweep results disclosed.
- Release: code, tokenizer, all E1 checkpoints, wandb logs, eval commit hash.

## 8. Compute budget

hours per run = tokens / (tok_s × 3600), with tok_s read off the first real
run. At an assumed 3M tok/s for the 24M-param arms:

| Block | Tokens | Node-hours @3M tok/s |
|---|---|---|
| E0 LR sweeps | 60B | ~6 |
| E1 headline + seeds | 840B | ~78 |
| E2 optimizer 2×2 | 60B | ~6 |
| E3 ablations | 280B | ~26 |
| E5 data scaling | 240B | ~22 |
| E1 scaling ladder (4 sizes × 2 arch × 30B) | 240B | ~30 (larger models slower) |
| **Total** | **~1.7T** | **~7–10 node-days** |

Trim path if needed: drop FFN-isoD and the xl ladder point, halve E3 to 10B
(→ ~5 node-days).

## 9. Decision gates and fallbacks

- **Gate after E0+E2 (≈1 day):** if SAN+muon is not within striking distance
  of FFN-isoP at 30B tokens, stop and diagnose before spending the headline
  budget. If the gap is architectural (survives LR sweep, all stabilizer
  ablations), the paper pivots to the honest characterization: *where* the
  FLOPs/params trade-off lands + H2 decomposition + H3 mechanism. That paper
  still exists; it's the claim strength that moves.
- **H2 null (gap uniform across exercises/regions):** drop the
  routing-vs-storage framing; keep dispensability + trainability.
- **Gated ≈ ReZero:** expected; fold into recipe.
- **SAN wins only at iso-FLOP, not iso-param:** report both honestly; the
  iso-FLOP framing is legitimate (attention FLOPs are cheap on modern HW —
  cite the wall-clock tok/s parity we measure directly).

## 10. Priority order

1. E0 LR sweeps → lock LRs (read tok_s off these runs to fill §8).
2. E2 2×2 at 30B — first science, gates the rest.
3. E1 headline pair (SAN, FFN-isoP) at 105B + E4/E6/E7 evals on checkpoints.
4. E3 ablations + E1 remaining matchings (isoF, isoD) + scaling ladder.
5. E5 data scaling.
6. Writing: intro around H1/H2/H3; Monad/Baguettotron as external anchors.
