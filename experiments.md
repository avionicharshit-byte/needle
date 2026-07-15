# Simple Attention Networks — Experiment Design

Target: NeurIPS main track. Hardware: one 8×H100 node. Data: PleIAs/SYNTH
(~78M docs ≈ 68B tokens, ChatML + reasoning traces, 16k BPE, seq_len 2048,
doc-masked packing). Theory and predictions P1–P8: `theory.md`.

## 1. Hypotheses

- **H1 — dispensability:** attention-only matches a standard transformer at
  equal parameter count on SYNTH val loss and downstream evals.
- **H2 — gap decomposition:** any SAN–FFN gap concentrates on
  parametric-knowledge tokens; measured by `exercise` type × token region
  (query/trace/answer via marker ids 4–7). Prediction: gap largest on
  memorization-answers, smallest on traces.
- **H3 — trainability:** (a) gates + 1/√(2N) init + ZCRMSNorm keep deep
  attention-only stacks trainable (depth ladder); (b) Muon matters *more*
  for SAN than for the FFN control (interaction, not main effect).

## 2. Fixed frame (identical across ALL arms)

- Tokenizer, seq_len 2048, packing, loss masking, z-loss, WSD shape,
  warmup/decay ratios, data order per comparison, val set, eval cadence.
- Global batch fixed: 64/device × 8 = 512 rows ≈ 1.05M tokens/step. Not a
  tuning knob; if it ever changes, E0 reruns.
- LR: Adam `--lr × 8`, Muon `--muon-lr × √8`; WD 0.01 on kernels, both arms.
- Params reported non-embedding first.
- Headline budget 105B tokens (~1.5 SYNTH passes; inside PleIAs' convergence
  recipe). Deliberately over-trained, not Chinchilla: H1/H2 need the
  capacity-limited regime. Ladder is iso-token (30B/size).

## 3. Configurations (measured, vocab 16,384)

| ID | Config | Total | Non-emb | Train GFLOPs/tok | Role |
|---|---|---|---|---|---|
| SAN | 20L d512, no FFN | 24.13M | 15.74M | ~0.40 | ours |
| FFN-isoP | 4L d512 ff2048 | 24.12M | 15.73M | ~0.20 | iso-parameter (Δ<0.04%) |
| FFN-isoF | 9L d512 ff2048 | ~43M | ~35M | ~0.39 | iso-training-FLOPs |
| FFN-isoD | 20L d512 ff2048 | 87.06M | 78.67M | ~0.72 | iso-depth |

Report all three matchings. Iso-param confounds no-FFN with depth (4L→20L);
FFN-isoD + the E3 depth ladder separate the two. Contingency (on demand):
iso-param+iso-depth = 20L FFN thinned to ~24M (d224).

Scaling ladder (FFN partner = same d, L/5 — FFN blocks are 5× attn blocks):

| Size | SAN | Total | Non-emb | FFN-isoP partner |
|---|---|---|---|---|
| tiny | 10L d256 | 6.16M | 1.97M | 2L d256 |
| small | 14L d384 | 12.49M | 6.20M | 3L d384 (+7%) |
| base | 20L d512 | 24.13M | 15.74M | 4L d512 (exact) |
| large | 26L d640 | 42.46M | 31.97M | 5L d640 (−4%) |
| xl | 32L d768 | 69.24M | 56.65M | 6L d768 (−6%) |

Ladder LR convention: all sizes reuse E0's base-tuned LRs (identical LR
within each pair → the gap is internally fair; Muon transfers LR across
width well). Before interpreting curve shape: 2-point LR probe at tiny and
xl (~2.5B each) to confirm base LR is in the flat region; per-size sweeps
only if it isn't.

```bash
# ladder, 30B each; base cells reused from E2
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

Steps ≈ tokens/1.05M (5B→5k, 20B→20k, 30B→30k, 105B→100k). Append to every
pretrain command: `--data-dir <corpus> --upload-checkpoints` (HF is the
source of truth; resume = same command + `--checkpoint checkpoints/<name>.pkl`).
Arms: SAN = defaults; FFN-isoP/isoF/isoD = `--ffn --num-layers 4/9/20`.

### E0 — LR fairness sweep (12 × 5B = 60B)
Dominant LR per arm, 3 points; winners carry into all commands below.

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

### E1 — Headline (8 × 105B = 840B)
Four matchings at seed 42 + 2 extra seeds for {SAN, FFN-isoP}. Primary
metric: val loss/PPL.

```bash
for seed in 42 43 44; do
  san pretrain --max-steps 100000 --seed $seed --log-rank-every 2500 \
      --wandb --name san_muon_base_105B_s$seed
  san pretrain --ffn --num-layers 4 --max-steps 100000 --seed $seed --log-rank-every 2500 \
      --wandb --name ffnisop_muon_base_105B_s$seed
done
san pretrain --ffn --num-layers 9  --max-steps 100000 --seed 42 --log-rank-every 2500 \
    --wandb --name ffnisof_muon_base_105B_s42
san pretrain --ffn --num-layers 20 --max-steps 100000 --seed 42 --log-rank-every 2500 \
    --wandb --name ffnisod_muon_base_105B_s42
```

### E2 — Optimizer×architecture 2×2 (4 × 30B = 120B; runs FIRST, gates E1)
Fresh cells (E1's different WSD horizon makes its 30B mark incomparable).
Prediction P6: adamw hurts SAN more. Muon cells reused by E5 + ladder.

```bash
for opt in muon adamw; do
  san pretrain --optimizer $opt --max-steps 30000 --log-rank-every 2500 \
      --wandb --name san_${opt}_base_31B_s42
  san pretrain --ffn --num-layers 4 --optimizer $opt --max-steps 30000 --log-rank-every 2500 \
      --wandb --name ffnisop_${opt}_base_31B_s42
done
```

### E3 — Component ablations (15 × 20B = 300B, SAN arm + FFN gate-mirror)
| Axis | Variants | Needs code |
|---|---|---|
| Residual | gated · ReZero · standard · none | flag |
| Norm | ZCRMSNorm · RMSNorm γ=1 · no QK-norm | flag |
| Post-attn norm | off · sandwich | flag |
| Depth iso-param | {8,20,32,48}L (d 800/512/400/320) × {gated, standard} | configs |

Seeding rule: cells with |Δ| within ~2× the E1 seed-noise band get 3 seeds
before interpretation.

```bash
# variant flags (--residual/--norm/--qk-norm/--post-attn-norm) land with §6 infra
for r in rezero standard none; do
  san pretrain --residual $r --max-steps 20000 --log-rank-every 2500 \
      --wandb --name e3_san_res-${r}_21B_s42
done
san pretrain --norm rms       --max-steps 20000 --wandb --name e3_san_rmsnorm_21B_s42
san pretrain --no-qk-norm     --max-steps 20000 --wandb --name e3_san_noqknorm_21B_s42
san pretrain --post-attn-norm --max-steps 20000 --wandb --name e3_san_sandwich_21B_s42

for cfg in "8 800" "32 400" "48 320"; do
  set -- $cfg
  san pretrain --num-layers $1 --d-model $2 --max-steps 20000 --log-rank-every 2500 \
      --wandb --name e3_san_depth${1}_21B_s42
  san pretrain --num-layers $1 --d-model $2 --residual standard --max-steps 20000 \
      --log-rank-every 2500 --wandb --name e3_san_depth${1}_nogate_21B_s42
done
san pretrain --residual standard --max-steps 20000 --log-rank-every 2500 \
    --wandb --name e3_san_depth20_nogate_21B_s42

# gate × architecture mirror (P4 interaction form)
san pretrain --ffn --num-layers 20 --max-steps 20000 --log-rank-every 2500 \
    --wandb --name e3_ffnisod_gated_21B_s42
san pretrain --ffn --num-layers 20 --residual standard --max-steps 20000 \
    --log-rank-every 2500 --wandb --name e3_ffnisod_nogate_21B_s42
```

### E4 — Gap decomposition (H2; evals on E1/E2 checkpoints, no training)
Per-exercise val sets × per-region (query/trace/answer) Δloss heatmap.

```bash
for ckpt in san_muon_base_105B_s42 ffnisop_muon_base_105B_s42; do
  san eval --checkpoint checkpoints/${ckpt}.pkl --by-exercise --by-region
done
```

### E5 — Data scaling (6 new × 30B = 180B; full-data cells from E2)
`--max-docs` ∈ {2M, 8M, 32M, full} ≈ {17×, 4×, 1×, 0.4×} repetition. P8:
FFN gains more from repetition.

```bash
for docs in 2000000 8000000 32000000; do
  san pretrain --max-docs $docs --max-steps 30000 \
      --wandb --name e5_san_muon_${docs}docs_31B_s42
  san pretrain --ffn --num-layers 4 --max-docs $docs --max-steps 30000 \
      --wandb --name e5_ffnisop_muon_${docs}docs_31B_s42
done
```

### E6 — Downstream evals (E1 checkpoints; needs lm-eval adapter)
0-shot loglikelihood: lambada, hellaswag, arc_easy, piqa, sciq, winogrande,
mmlu — PleIAs protocol, compared against Monad-56M / Baguettotron-321M task
accuracies (never their losses: different tokenizer).

```bash
for ckpt in checkpoints/*_105B_*.pkl; do
  san eval --checkpoint $ckpt \
      --tasks lambada hellaswag arc_easy piqa sciq winogrande mmlu
done
```

### E7 — Mechanistic (H3; no new training)
Rank + gate trajectories from wandb; SV spectra over milestone checkpoints
(0/25/50/75/100%). Money figure: end-of-training per-layer rank for
{SAN+muon, SAN+adamw, SAN-no-gates, FFN-isoP}.

```bash
python scripts/sv_spectra.py checkpoints/san_muon_base_105B_s42_step*.pkl \
                             checkpoints/san_adamw_base_31B_s42_step*.pkl
```

## 5. Figures

1. Val loss vs tokens: SAN vs three matchings (E1).
2. H2 heatmap: Δloss by exercise × region (E4).
3. Scaling: val loss vs non-emb params, both arms (ladder).
4. Rank-vs-depth panel (E7).
5. Gate trajectories heatmap (layers × time).
6. Optimizer×architecture interaction bars (E2).
7. Downstream table vs Monad/Baguettotron (E6).

## 6. Infrastructure to build

| Need | Size |
|---|---|
| Residual/norm variant flags (ReZero, standard, RMSNorm, no-QK, sandwich) | small, architecture.py |
| Per-exercise val sets + per-region loss slicing | small, data.py + eval.py |
| lm-eval loglikelihood adapter (stub in eval.py) | ~2 days |
| SV-spectra script | small |
| Param/FLOP calculator script | trivial |

Run naming: `{arch}_{opt}_{size}_{tokens}_{seed}`, e.g. `san_muon_base_105B_s42`.

## 7. Statistical methodology

- 3 seeds headline pair; ablation cells 1 seed, upgraded to 3 when |Δ| is
  within ~2× the seed-noise band.
- Same data seed across arms per comparison; val set fixed.
- Mean ± 95% CI; paired-by-block bootstrap for Δloss claims.
- All hyperparameters + E0 results in appendix. Release code, tokenizer,
  E1 checkpoints, wandb logs.

## 8. Decision gates

- **After E0+E2 (~1 day):** SAN+muon not near FFN-isoP at 30B → stop,
  diagnose; if architectural, pivot to trade-off characterization + H2 + H3.
- **H2 null:** drop routing-vs-storage framing; keep H1 + H3.
- **Gated ≈ ReZero:** expected; recipe, not novelty.
- **SAN wins only iso-FLOP:** report both matchings honestly.

## 9. Priority order

1. E0 (read tok_s → re-pin §8).
2. E2 → gate decision.
3. E1 headline pair + E4/E6/E7 evals.
4. E3 + isoF/isoD + ladder.
5. E5.
6. Writing.

## 10. Results log

**2026-07-15 — infra:** cudnn attention was silently falling back to unfused
XLA (auto dispatcher never tries cudnn with a mask); explicit
`implementation="cudnn"` accepted the doc mask: 6.5× on the op, 2.75×
end-to-end → base config 2.3M tok/s (§8 re-pinned). Corpus tokenized:
68.33B tokens / 136.7GB, uploading to HF.

**2026-07-15 — infra:** remat is mandatory at 20L/batch 64/seq 2048 (~70GiB
peak without, even with fused attention) — removing it OOM'd the first
extension cell; re-enabled permanently. Throughput numbers already included it.

**2026-07-15 — E0 complete incl. extensions (15 cells, val/loss @5k):**

| Arm | 0.5× | 1× | 2× | 4× ext | 8× ext | Verdict |
|---|---|---|---|---|---|---|
| SAN muon | 2.248 | **2.245** | 2.251 | — | — | **LOCKED 0.02** (interior; landscape flat 2.245–2.251 across 4×) |
| FFN muon | 2.223 | 2.224 | **2.216** | 2.237 (turned) | — | **LOCKED 0.04** (interior after extension; 0.5×/1× inversion ~0.001 = noise) |
| SAN adamw | 2.361 | 2.293 | **2.267** | 2.321 (turned) | — | **LOCKED 6e-4** (interior after extension) |
| FFN adamw | 2.305 | 2.254 | 2.230 | **2.219** (still ↓, decelerating) | running | pending 2.4e-3 |

- P6-direction note strengthened at tuned LRs (still unseeded/5B): muon−adamw
  gap ≈ 0.026 on SAN vs ≈ 0.003 on FFN — the predicted interaction shape.

- Default LRs were centered too low for adamw arms; sweep working as intended.
- Preliminary P6-direction note (unseeded, 5B tokens — not evidence): muon−adamw
  gap larger for SAN (0.022) than FFN (0.014).
- Best-FFN leads best-SAN by ~0.03 nats at 5B — expected at data-limited scale;
  E2@30B is the first real comparison.
- Ladder ran early by accident: san_tiny complete (valid), san_small died at
  13.5k (HF streaming reset — memmap corpus eliminates this class), san_large
  killed. FFN ladder cells deliberately held until E0 LRs lock.
