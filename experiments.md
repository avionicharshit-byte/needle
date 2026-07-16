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
width well). Before interpreting curve shape: 3-point LR probe (0.5×/1×/2×)
at tiny and xl (2.5k-step horizon, ~2.6B each — a 30k-horizon run's step-2500
val is not comparable) to confirm base LR is in the flat region; per-size
sweeps only if it isn't.

As-run: 7×H100 frame-matched (batch 73 = 511 rows, LR flags as E3), probes
√7-corrected; base pair rerun on-node (`ladder_{san,ffnisop}_base_31B_s42`)
so the curve is internally consistent — E2's 512-row cells are not mixed
into the ladder figure.

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
LRs default to the E0 locks per arm (SAN/FFN × muon/adamw, resolved at
runtime from the config; muon runs keep adam side at 3e-4 as swept) —
commands need no LR flags; explicit flags override. isoF/isoD inherit the
isoP locks (stated convention).

### E0 — LR fairness sweep (CLOSED 2026-07-15; 17 × 5B = 85B)
3-point sweep per arm at 0.5×/1×/2× (muon 1× = 0.02, adamw 1× = 3e-4);
boundary rule: sweep-edge winner gets one extension per round until the
curve turns. Ran pre-corpus on the HF streaming path; runs
`e0_{arm}_{opt}_(m)lr<value>`, 5k steps, explicit LR flags.

Val/loss @5k:

| Arm | 0.5× | 1× | 2× | 4× | 8× | 16× | Lock |
|---|---|---|---|---|---|---|---|
| SAN muon | 2.248 | **2.245** | 2.251 | | | | **0.02** |
| FFN muon | 2.223 | 2.224 | **2.216** | 2.237 | | | **0.04** |
| SAN adamw | 2.361 | 2.293 | **2.267** | 2.321 | | | **6e-4** |
| FFN adamw | 2.305 | 2.254 | 2.230 | 2.219 | **2.199** | 2.216 | **2.4e-3** |

- All four locks interior (loss turns on both sides); they are now the CLI
  defaults (see preamble). SAN-muon landscape flat: 2.245–2.251 across 4×.
- P6 signal at 5B — full crossover: SAN prefers muon (2.245 vs 2.267,
  Δ0.022); FFN prefers adamw (2.199 vs 2.216, Δ0.017).
- FFN @4.8e-3 (16×): ALL FFN gates collapsed (0.02–0.18, mean 0.07) while
  attn gates stayed structured — under LR stress the standard transformer
  self-pruned toward attention-only and still hit 2.216. H1-flavored, from
  an undesigned direction (caveat: gated off ≠ removed).

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

### E2 — Optimizer×architecture 2×2 (CLOSED 2026-07-16; 4 × 31.5B = 126B)
Fresh cells at locked LRs (E1's different WSD horizon makes its 30B mark
incomparable). Runs `{arch}_{opt}_base_31B_s42`, 30k steps, memmap corpus,
8×H100. Muon cells reused by E5 + ladder.

| Cell | val@30k | ppl | tok/s | wall |
|---|---|---|---|---|
| SAN muon | 2.1126 | 8.27 | 2.0M | 4h31m |
| FFN-isoP muon | **2.0933** | 8.11 | 4.4M | 2h08m |
| SAN adamw | 2.1103 | 8.25 | 2.0M | 4h28m |
| FFN-isoP adamw | 2.0950 | 8.13 | 4.4M | 2h06m |

- **GATE PASSED — E1 GO.** Best-SAN − best-FFN = 0.017 nats (0.8%) at 30B,
  narrowed from 0.046 at 5B (E0). Muon pair: 0.019.
- P6 at 30B: the 5B crossover washed out — within-arch optimizer deltas
  ≤0.002 nats (SAN: adamw −0.002; FFN: muon −0.002), noise-scale, both
  directions flipped vs E0. Interaction is a short-horizon phenomenon here.
- Gate dynamics (wandb): optimizer dominates gate fate at tuned LR. Muon
  suppresses — FFN cell ffn gates end 0.06/0.09/0.16/0.39 (mean 0.17,
  echoing the E0 4.8e-3 collapse at the *locked* LR), attn L0 0.03;
  SAN-muon early layers drift to 0.17–0.33, layers 14–17 rise to 0.65–0.81.
  AdamW saturates — FFN cell ffn L1–3 → 1.0, attn L2–3 → 1.0; SAN-adamw
  layers 10–17 end 0.82–0.99.
- SAN-adamw early instability: val 3.53@500 → 3.78@1000 before recovering;
  muon cells monotone.
- Infra: three transient HF 504s — ffnisop_muon 75% milestone (22500) and
  final upload failed (re-uploaded from pod), san_adamw step0 milestone
  failed; benign core dump at san_adamw teardown after successful upload.

### E3 — Component ablations (16 × 21B = 336B, SAN arm + FFN gate-mirror)
Includes the gated 20k baseline `e3_san_gated_21B_s42` — E2's SAN cell is a
30k-WSD run, incomparable at the 20k horizon. Depth cells d800/d400 use
10H/5KV: cuDNN flash needs head_dim % 8 = 0 (100/50 rejected); KV ratio 2
keeps attention params unchanged. Non-8-GPU nodes: frame-match batch
(512/n per device) + explicit LRs canceling device scaling. As-run:
7×H100, `--batch-size 73` (511 rows), `--lr 3.4286e-4`, `--muon-lr
0.021381` SAN / `0.042762` FFN (effective = the 8-device locks). Blackwell
(2026-07-16, returned): cuDNN doc-mask engine has no fast path — 5.25ms vs
1.07ms is_causal at B171 ⇒ ~1.15× H100/GPU on this workload; avoid.
| Axis | Variants | Needs code |
|---|---|---|
| Residual | gated · ReZero · standard · none | flag |
| Norm | ZCRMSNorm · RMSNorm γ=1 · no QK-norm | flag |
| Post-attn norm | off · sandwich | flag |
| Depth iso-param | {8,20,32,48}L (d 800/512/400/320) × {gated, standard} | configs |

Seeding rule: cells with |Δ| within ~2× the E1 seed-noise band get 3 seeds
before interpretation.

Note: with WD on kernels only (fixed frame), `--norm rms` is
optimizer-equivalent to zcrms (γ = 1+γ_zc, shift-invariant Adam) — the cell
is a Δ≈0 noise control, not a real ablation, unless WD is extended to norm
scales (which would break the fixed frame).

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
| ~~Residual/norm variant flags~~ DONE 2026-07-16 (`--residual {gated,rezero,standard,none}`, `--norm {zcrms,rms}`, `--no-qk-norm`, `--post-attn-norm`) | architecture.py |
| ~~Per-exercise/per-region loss slicing~~ DONE 2026-07-17 (`san eval --by-exercise --by-region [--group-field language] [--val-docs N]`; whole-doc packing, labels from marker ids, validated on E2 ckpt) | eval.py |
| ~~lm-eval loglikelihood adapter~~ DONE 2026-07-17 (`san eval --checkpoint X --tasks sciq …`; needs `pip install lm-eval` on the eval pod; validated: sciq acc 0.70 @limit 20 on E2 SAN ckpt) | lm_eval_adapter.py |
| ~~SV-spectra script~~ DONE 2026-07-17 (`python scripts/sv_spectra.py <ckpts> --out spectra`; per-kernel SVs + effective/stable rank to npz) | scripts/sv_spectra.py |
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

1. E4/E6/E7 evals.

## 10. Results log

- **2026-07-15 — E0 CLOSED.** Locks: SAN muon 0.02 · FFN muon 0.04 ·
  SAN adamw 6e-4 · FFN adamw 2.4e-3. Full table + findings: §4 E0.
- **2026-07-16 — E2 CLOSED, GATE PASSED.** SAN within 0.017 nats of
  FFN-isoP at 30B (was 0.046 at 5B); P6 crossover gone at 30B → E1 GO.
  Full table + findings: §4 E2.
