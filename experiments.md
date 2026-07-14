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
> **H3 — trainability.** Deep attention-only stacks need the right
> stabilizers, for two distinct reasons (theory.md §5): (a) [derived] gated
> residuals + 1/√(2N) init + ZCRMSNorm keep residual-stream variance bounded
> independent of depth — remove them and the depth ladder should degrade;
> (b) [hypothesis] without FFNs the per-token pathway is a deep product of
> linear maps, so spectrally imbalanced (AdamW-style) updates compound —
> Muon's flat-spectrum updates should matter *more* for SAN than for the FFN
> control (an interaction, not a main effect). Note: Dong et al.'s rank
> collapse does NOT predict SAN fails — residuals prevent it (their own
> result); we measure per-layer rank to test what remains.

Positioning: vs Dong et al. (their theorem licenses residuals, not FFNs — we
test the remainder), Geva et al. (H2 hypothesis source), ReZero/SkipInit/
LayerScale (gated residual family — recipe, not novelty), Gemma-style
zero-centered norm gain (recipe), Muon (novel angle: enabler of FFN-free
depth), gMLP (mirror-image evidence). PleIAs Monad-56M / Baguettotron-321M —
same data, same format, published numbers = free external reference points.
Theory→experiment mapping: predictions P1–P8 in theory.md §7.

## 2. Fixed experimental frame (identical across ALL arms)

- Tokenizer (16,384 BPE, trained on ChatML-formatted SYNTH), seq_len 2048,
  packing, loss masking, z-loss, WSD schedule shape, warmup/decay ratios,
  data order (same `--seed` within a comparison), val set (seeded window
  subsample, disjoint from train), eval cadence.
- **Global batch: fixed at 64/device × 8 = 512 rows ≈ 1.05M tokens for every
  run in the paper.** Batch size sets gradient noise and steps-per-token, so
  it is part of the frame, not a utilization knob — no per-arm tuning, even
  though light arms could fit more. It must fit the heaviest configs
  (FFN-isoD, 48L depth cell), and E0's LRs are only valid at this batch: if
  it ever changes, E0 reruns. (If a future config OOMs, the fix is gradient
  accumulation — same global batch, smaller microbatch — not a batch change.)
- LR conventions: Adam `--lr × 8`, Muon `--muon-lr × √8`. Weight decay 0.01
  on Dense kernels in both optimizer arms.
- **Params reported as non-embedding first** — tied 16k×d embeddings are
  identical within every pair and would otherwise flatter small models.
- Default token budget: 100k steps × 512 rows × 2048 = **105B tokens**
  (~1.5 passes over SYNTH; PleIAs report convergence at 100–200B).
- **Why not Chinchilla-optimal (~0.5B tokens at 24M):** Chinchilla allocates
  N vs D under a free model size at fixed compute; our N is fixed by the
  research question, and modern small models are deliberately over-trained
  (10³–10⁴ tokens/param). Decisively: H1/H2 test whether FFN *capacity*
  matters, and capacity only binds in the over-trained, capacity-limited
  regime — at Chinchilla scale the FFN memories are barely loaded and SAN
  would "match" trivially (a false positive for our own hypothesis). The
  105B headline also sits inside PleIAs' reported convergence recipe, keeping
  Monad/Baguettotron anchors meaningful. The scaling ladder is iso-token
  (30B at every size), not Chinchilla-allocated, so size is the only moving
  variable.

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

Known confound and its coverage: iso-param changes two things at once (drops
FFNs AND deepens 4L→20L). FFN-isoD (equal depth, FFN's marginal value) plus
the E3 depth ladder (SAN quality vs depth at fixed params) jointly separate
depth from FFN effects. Contingency cell if reviewers press: iso-param AND
iso-depth — 20L FFN thinned to ~24M total (≈ d224, d_ff 4d, 4 heads) —
unbudgeted, ~20B-token tier, run on demand.

Scaling ladder (SAN side; FFN-isoP partners computed the same way):

| Size | Config | Total | Non-emb | FFN-isoP partner (same d; L≈SAN_L/5) |
|---|---|---|---|---|
| tiny | 10L d256 | 6.16M | 1.97M | 2L d256 |
| small | 14L d384 | 12.49M | 6.20M | 3L d384 (+7%) |
| base | 20L d512 | 24.13M | 15.74M | 4L d512 (exact) |
| large | 26L d640 | 42.46M | 31.97M | 5L d640 (−4%) |
| xl | 32L d768 | 69.24M | 56.65M | 6L d768 (−6%) |

The L/5 partner rule is exact because FFN blocks are 5× attention blocks
(15d² vs 3d²) at d_ff=4d, GQA 2:1. Report exact counts where rounding bites.

Ladder commands (30B tokens each; base cells reused from E2):

```bash
for cfg in "tiny 10 256 4 2" "small 14 384 6 3" "large 26 640 10 5" "xl 32 768 12 6"; do
  set -- $cfg
  san pretrain --num-layers $2 --d-model $3 --num-heads $4 --num-kv-heads $5 \
      --max-steps 30000 --wandb --name ladder_san_${1}_31B_s42
done
for cfg in "tiny 2 256 4 2" "small 3 384 6 3" "large 5 640 10 5" "xl 6 768 12 6"; do
  set -- $cfg
  san pretrain --ffn --num-layers $2 --d-model $3 --num-heads $4 --num-kv-heads $5 \
      --max-steps 30000 --wandb --name ladder_ffnisop_${1}_31B_s42
done
```

## 4. Experiment matrix

All commands assume the 8-GPU node with defaults `--batch-size 64
--seq-len 2048` → ~1.05M tokens/step, so steps ≈ tokens/1.05M (5B→5k steps,
20B→20k, 30B→30k, 105B→100k). Runs execute sequentially (each saturates the
node). Training runs that feed E7 carry `--log-rank-every 2500`.
Once the pre-tokenized corpus exists (`san tokenize-corpus`, one-time),
append `--data-dir /workspace/data/synth` to every pretrain command —
faster input path, exact global shuffle, no HF dependence mid-run.
Arm shorthands: SAN = defaults; FFN-isoP = `--ffn --num-layers 4`;
FFN-isoF = `--ffn --num-layers 9`; FFN-isoD = `--ffn --num-layers 20`.

### E0 — LR fairness sweep (before anything headline)
Per arm {SAN, FFN-isoP} × {muon, adamw}: 3-point sweep (0.5×, 1×, 2× default)
of the *dominant* LR per cell — `--muon-lr` in muon cells (adam side fixed),
`--lr` in adamw cells — at 5B tokens each. Best LR per arm carries forward.
"We tuned ours and not the control" is a desk-reject risk; this is the
insurance. 12 runs × 5B = 60B tokens.

```bash
for mlr in 0.01 0.02 0.04; do
  san pretrain --optimizer muon --muon-lr $mlr --max-steps 5000 \
      --wandb --name e0_san_muon_mlr$mlr
  san pretrain --ffn --num-layers 4 --optimizer muon --muon-lr $mlr --max-steps 5000 \
      --wandb --name e0_ffnisop_muon_mlr$mlr
done
for lr in 1.5e-4 3e-4 6e-4; do
  san pretrain --optimizer adamw --lr $lr --max-steps 5000 \
      --wandb --name e0_san_adamw_lr$lr
  san pretrain --ffn --num-layers 4 --optimizer adamw --lr $lr --max-steps 5000 \
      --wandb --name e0_ffnisop_adamw_lr$lr
done
```
Winner per arm = lowest val loss at 5k steps; carry into every command below
(append `--muon-lr <best>` / `--lr <best>`; omitted here for brevity).

### E1 — Headline comparison
8 runs × 105B tokens: the four matchings {SAN, FFN-isoP, FFN-isoF, FFN-isoD}
at seed 42 (muon, E0-tuned LRs), plus 2 extra seeds each for the load-bearing
pair {SAN, FFN-isoP}. = 840B tokens.
Primary metric: val loss/PPL; secondary: downstream evals (E6), throughput.

```bash
for seed in 42 43 44; do
  san pretrain --max-steps 100000 --seed $seed --log-rank-every 2500 \
      --wandb --upload-checkpoints --name san_muon_base_105B_s$seed
  san pretrain --ffn --num-layers 4 --max-steps 100000 --seed $seed --log-rank-every 2500 \
      --wandb --upload-checkpoints --name ffnisop_muon_base_105B_s$seed
done
san pretrain --ffn --num-layers 9  --max-steps 100000 --seed 42 --log-rank-every 2500 \
    --wandb --upload-checkpoints --name ffnisof_muon_base_105B_s42
san pretrain --ffn --num-layers 20 --max-steps 100000 --seed 42 --log-rank-every 2500 \
    --wandb --upload-checkpoints --name ffnisod_muon_base_105B_s42
```

### E2 — The optimizer×architecture 2×2 (H3 core; runs FIRST, gates E1)
All four cells {muon, adamw} × {SAN, FFN-isoP} at 30B tokens, fresh runs —
the muon cells cannot be borrowed from E1 (E1 runs later, and its 105B WSD
horizon puts a different LR at the 30B mark). The prediction that matters:
adamw hurts the SAN arm *more* than the FFN arm (interaction term, not main
effect — theory.md P6). Flag-complete: `--optimizer`, `--ffn`.
4 cells × 30B = 120B tokens. The two muon cells are REUSED as the E5
full-data cells and the ladder base-size cells.

```bash
for opt in muon adamw; do
  san pretrain --optimizer $opt --max-steps 30000 --log-rank-every 2500 \
      --wandb --upload-checkpoints --name san_${opt}_base_31B_s42
  san pretrain --ffn --num-layers 4 --optimizer $opt --max-steps 30000 --log-rank-every 2500 \
      --wandb --upload-checkpoints --name ffnisop_${opt}_base_31B_s42
done
```

### E3 — Component ablations (20B tokens each, SAN arm)
| Axis | Variants | Needs code |
|---|---|---|
| Residual | gated (ours) · ReZero (α init 0, no sigmoid) · standard `x+F(x)` · none | flag |
| Norm | ZCRMSNorm (ours) · RMSNorm γ=1 · no QK-norm | flag |
| Post-attn norm | off (ours) · sandwich `x+σ(g)·Norm(Attn(Norm(x)))` | flag |
| Depth at iso-param | {8, 20, 32, 48}L (d adjusted, e.g. 48L→d320) × {gates on/off} | configs only |
~15 new runs × 20B = 300B tokens (gated-20L SAN is the default, not re-run;
includes the 2-run FFN mirror for the gate×architecture interaction —
theory.md P4's interaction form, mirroring what E2 does for the optimizer axis).
Report one table: Δval-loss per component. Seeding rule (also §7): any cell
whose |Δ| is within ~2× the seed-noise band (from E1's 3-seed runs) gets
upgraded to 3 seeds before being interpreted; large-effect cells don't need it.

```bash
# residual / norm variants (--residual, --norm, --qk-norm, --post-attn-norm
# flags land with the E3 infra, §6)
for r in rezero standard none; do
  san pretrain --residual $r --max-steps 20000 --log-rank-every 2500 \
      --wandb --name e3_san_res-${r}_21B_s42
done
san pretrain --norm rms      --max-steps 20000 --wandb --name e3_san_rmsnorm_21B_s42
san pretrain --no-qk-norm    --max-steps 20000 --wandb --name e3_san_noqknorm_21B_s42
san pretrain --post-attn-norm --max-steps 20000 --wandb --name e3_san_sandwich_21B_s42

# depth at iso-param (~15.7M non-emb): 8L d800 / 32L d400 / 48L d320,
# each with gated (default) and standard residual (= gates off)
for cfg in "8 800" "32 400" "48 320"; do
  set -- $cfg
  san pretrain --num-layers $1 --d-model $2 --max-steps 20000 --log-rank-every 2500 \
      --wandb --name e3_san_depth${1}_21B_s42
  san pretrain --num-layers $1 --d-model $2 --residual standard --max-steps 20000 \
      --log-rank-every 2500 --wandb --name e3_san_depth${1}_nogate_21B_s42
done
san pretrain --residual standard --max-steps 20000 --log-rank-every 2500 \
    --wandb --name e3_san_depth20_nogate_21B_s42   # 20L gated = default run, reused

# gate × architecture interaction at matched depth (the FFN mirror — without
# it, "gates matter more for attention-only" is unsupported): FFN-isoD 20L
# with and without gates. SAN cells come from the rows above.
san pretrain --ffn --num-layers 20 --max-steps 20000 --log-rank-every 2500 \
    --wandb --name e3_ffnisod_gated_21B_s42
san pretrain --ffn --num-layers 20 --residual standard --max-steps 20000 --log-rank-every 2500 \
    --wandb --name e3_ffnisod_nogate_21B_s42
```

### E4 — Gap decomposition (H2; evaluation of E1/E2 checkpoints — no new training)
- Per-exercise val loss: separate packed val sets filtered by `exercise`
  (memorization / mcq / math / rag / creative-writing).
- Per-region val loss: split token positions by `<think>`…`</think>` marker
  ids (4–7) into query / trace / answer regions.
- The H2 figure: SAN−FFN Δloss per (exercise × region) cell, with the
  memorization-answer cell predicted worst and trace cells predicted best.

```bash
# per-exercise + per-region eval flags land with the E4 infra (§6)
for ckpt in san_muon_base_105B_s42 ffnisop_muon_base_105B_s42; do
  san eval --checkpoint checkpoints/${ckpt}.pkl --by-exercise --by-region
done
```

### E5 — Data scaling / repetition (uses `--max-docs`)
SAN + FFN-isoP at 30B tokens with `--max-docs` ∈ {2M, 8M, 32M, full}
(≈17×, 4×, 1×, 0.4× repetition). Secondary H2 probe (theory.md P8): FFN arms
should benefit more from repetition (memorization capacity), SAN should
degrade less under data constraint. Full-data cells reused from E2 → 6 new
runs × 30B = 180B tokens.

```bash
for docs in 2000000 8000000 32000000; do
  san pretrain --max-docs $docs --max-steps 30000 \
      --wandb --name e5_san_muon_${docs}docs_31B_s42
  san pretrain --ffn --num-layers 4 --max-docs $docs --max-steps 30000 \
      --wandb --name e5_ffnisop_muon_${docs}docs_31B_s42
done
# full-data cells = san_muon_base_31B_s42 / ffnisop_muon_base_31B_s42 (E2)
```

### E6 — Downstream evals (no training; needs lm-eval adapter)
0-shot loglikelihood tasks suitable for <100M models: lambada, hellaswag,
arc_easy, piqa, sciq, winogrande, mmlu (expect near random floor at 24M —
report anyway, matching PleIAs' protocol and prompt format for direct
comparison against published Monad-56M / Baguettotron-321M numbers). Task
accuracies are cross-tokenizer comparable; val losses are NOT (different
vocab) — never compare loss against their models. Run on every E1 checkpoint.

```bash
# --tasks activates once the lm-eval adapter lands (§6)
for ckpt in checkpoints/*_105B_*.pkl; do
  san eval --checkpoint $ckpt \
      --tasks lambada hellaswag arc_easy piqa sciq winogrande mmlu
done
```

### E7 — Mechanistic (H3; instrumentation mostly built)
- Rank trajectories: `--log-rank-every 2500` on all E1/E2/E3 runs (free).
- Gate trajectories: per-layer σ(g) over training (already logged).
- Singular-value spectra of Q/K/V/O kernels: muon vs adamw arms, at 0/25/50/
  100% of training (offline script over checkpoints).
- Money figure: per-layer effective rank at end of training for {SAN+muon,
  SAN+adamw, SAN-no-gates, FFN-isoP} — Dong et al.'s prediction tested.

```bash
# rank + gate trajectories: already in wandb from --log-rank-every / gate logging
# SV spectra over milestone checkpoints (script lands with §6 infra):
python scripts/sv_spectra.py checkpoints/san_muon_base_105B_s42_step*.pkl \
                             checkpoints/san_adamw_base_31B_s42_step*.pkl
```

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
| ~~Milestone checkpoint retention~~ DONE — `<name>_step<k>.pkl` kept at 0/25/50/75% + final; grad-norm, LR, and token-count logging added; checkpoint meta carries commit/argv/batch/LRs | done |
| Residual/norm variant flags (ReZero, standard, RMSNorm, no-QK, sandwich) | small, architecture.py |
| Per-exercise val sets (filter by `exercise` before packing; rare exercises ≈1–2% of stream → build offline, expect a long scan) | small, data.py + eval.py |
| Per-region loss slicing (marker ids 4–7 → query/trace/answer masks) | small, eval.py |
| lm-eval loglikelihood adapter (stub exists in eval.py) | ~2 days |
| SV-spectra script over checkpoints | small |
| Multi-run launcher (shell loop over configs × seeds is fine) | trivial |
| Param/FLOP calculator (one-liner exists; commit as script) | trivial |

Run naming convention (wandb + checkpoints): `{arch}_{opt}_{size}_{tokens}_{seed}`,
e.g. `san_muon_base_105B_s42` — encode everything needed to regenerate the run.

## 7. Statistical methodology

- 3 seeds for the headline pair; 1 seed for ablation cells, upgraded to 3
  seeds whenever the cell's |Δ| falls within ~2× the headline seed-noise band
  (a conclusion may never rest on an unseeded within-noise difference).
- Same data seed across arms within a comparison; val set fixed (seed 3407).
- Mean ± 95% CI on val loss; paired-by-block bootstrap for Δloss claims.
- Every hyperparameter in the appendix; E0 sweep results disclosed.
- Release: code, tokenizer, all E1 checkpoints, wandb logs, eval commit hash.

## 8. Compute budget

hours per run = tokens / (tok_s × 3600). **Measured: 3.06M tok/s on the
tiny (6.2M) ladder cell** (2026-07-14) — note that is only ~4% MFU, i.e. the
tiny cell is pipeline-bound, so the 24M base arms may run slower (~1.5–3M
tok/s); re-pin this table from the first base-config run. Effective
throughput incl. eval/save pauses is ~7% below the instantaneous tok_s
(2.71 it/s × 1.05M ≈ 2.84M effective). Assuming ~3M for base:

| Block | Tokens | Node-hours @3.06M tok/s |
|---|---|---|
| E0 LR sweeps | 60B | ~5.5 |
| E1 headline + seeds | 840B | ~76 |
| E2 optimizer 2×2 (4 fresh cells) | 120B | ~11 |
| E3 ablations (incl. FFN gate-mirror cells) | 300B | ~27 |
| E5 data scaling (full-data cells reused from E2) | 180B | ~16 |
| Scaling ladder (base cells reused from E2; 6 new runs × 30B) | 180B | ~21 (larger models slower) |
| **Total** | **~1.68T** | **~157 h ≈ 6.5 node-days pure training** |

Plus ~8h of per-run overhead (val build, compile) across ~50 runs and the
deliberate analysis pause at the E0+E2 gate → **~8–10 calendar days**.

Trim path (tiered; tier 1 breaks nothing):
- **Tier 1 (−350B ≈ 21%):** drop the xl ladder pair (4-point curve suffices);
  E5 → 2 new runs (extreme-repetition point only; P8 becomes a two-point
  contrast); drop the sandwich-norm cell (maps to no prediction); run
  FFN-isoF/isoD at 30B instead of 105B — they then share E2's WSD horizon,
  so the four-matching triangle lives coherently at the 30B tier and the
  105B tier carries only the seeded headline pair.
- **Tier 2 (only if forced):** E3 at 10B (noisier Δs, more seed upgrades).
- **Never:** E0, E2, headline seeds, gate-mirror cells, E4/E6/E7 evals.
- Cells predicted to diverge (no-residual, deep no-gate) may be stopped once
  divergence is unambiguous — the divergence is the datapoint.

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
