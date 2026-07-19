# Simple Attention Networks — Experiment Design

Paper: **"A Controlled Study of Attention-Only Transformers"**.
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

**LADDER + PROBES CLOSED 2026-07-19.** Probes (val @2.5k, effective muon LR
0.5×/1×/2× of lock): tiny 2.8514/2.8558/2.8574 (flat, spread 0.006); xl
2.0739/**2.0656**/2.0783 (1× interior minimum) → base LR valid at both ends,
no per-size sweeps needed. Ladder val @30k (31.5B tokens each):

| Size | Non-emb | SAN | FFN-isoP | Δ (SAN−FFN) |
|---|---|---|---|---|
| tiny (10L/2L d256) | 1.97M | **2.7219** | 2.7664 | −0.0445 |
| small (14L/3L d384) | 6.20M | 2.3771 | **2.3697** | +0.0074 |
| base (20L/4L d512) | 15.74M | 2.1159 | **2.0963** | +0.0196 |
| large (26L/5L d640) | 31.97M | 1.9275 | **1.9059** | +0.0216 |
| xl (32L/6L d768) | 56.65M | 1.7849 | **1.7647** | +0.0202 |

- SAN WINS at tiny (the 2L FFN partner is too shallow — depth floor bites
  the FFN arm first); gap crosses zero by small and **plateaus at ~0.02
  nats from base upward** — no divergence with scale through 57M non-emb.
- Frame consistency: ladder base (511-row) 2.1159 vs E2 base (512-row)
  2.1126 — Δ0.0033 across frame + pod + data-order change.
- Figure: `assets/ladder_scaling.png`.

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

### E1 — Headline (8 × 105B = 840B; 6/8 done 2026-07-19)
Four matchings at seed 42 + 2 extra seeds for {SAN, FFN-isoP}. Primary
metric: val loss/PPL. Remaining: s44 pair (SAN @70k, FFN-isoP queued).

| Cell | Params | val@100k |
|---|---|---|
| SAN s42 | 24.13M | 2.0711 |
| SAN s43 | 24.13M | 2.0646 |
| FFN-isoP s42 | 24.12M | 2.0656 |
| FFN-isoP s43 | 24.12M | 2.1137 † |
| FFN-isoF (9L) | ~43M | 1.8059 |
| FFN-isoD (20L) | 87.06M | 1.5982 |

† **Terminal instability**: s43 tracked s42 to within 0.002 through step
99k (2.0757 vs 2.0741), then grad norm ramped 0.09→0.30 over the final
~800 steps at the LR floor and val bounced to 2.1137. Train loss degraded
in lockstep — a real optimization event, not val noise. No SAN run showed
this; rhymes with the FFN arm's E0 gate-collapse fragility. Report final
values as-is with this note; the 3-seed mean absorbs it.
(`assets/ffnisop_s43_tail.png`)

- Iso-param gap at 105B, seed 42: 0.0055 (SAN behind) — down from 0.019
  at 30B (E2). Seed 43 inverted by the instability. s44 decides.
- isoF/isoD anchor the capacity axis: 1.81 (43M) / 1.60 (87M) vs ~2.07
  (24M pair) — matchings behave as designed.
- Figure: `assets/e1_val_curves.png`.

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

**Interim results 2026-07-19 (13/16 cells; pending: depth48-nogate,
2 ffnisod mirrors).** Val @20k (21B), baseline gated 20L = **2.1343**:

| Cell | val | Δ vs gated |
|---|---|---|
| sandwich (post-attn norm) | **2.1251** | −0.0092 |
| standard residual | 2.1330 | −0.0013 |
| rezero | 2.1375 | +0.0032 |
| rmsnorm (equivalence control) | 2.1358 | +0.0015 |
| no QK-norm | **8.2758** | DIVERGED |
| no residual | 2.8826 | +0.75 |
| depth8 (d800) / nogate | 2.1680 / 2.1639 | +0.034 / +0.030 |
| depth32 (d400) / nogate | 2.1528 / 2.1503 | +0.019 / +0.016 |
| depth48 (d320) | 2.1751 | +0.041 |

- **QK-norm is load-bearing**: removing it diverges outright at the locked
  muon LR (8.28 ≈ uniform). The strongest single-component finding.
- **Gates are performance-neutral**: standard ≈ gated at 20L (−0.0013) and
  at every depth tested (nogate marginally better at 8L/32L). H3a reframes:
  trainability comes from norms + init; gates are a *diagnostic* (their
  self-pruning trajectories), not an accelerant. Matches the §8 expectation
  ("gated ≈ ReZero: recipe, not novelty").
- **Sandwich norm is the only variant that beats baseline** (−0.009,
  ~6× the noise floor) — candidate default for a camera-ready recipe.
- **rmsnorm Δ+0.0015 confirms the predicted optimizer-equivalence** and
  doubles as the same-seed noise-floor calibration: |Δ| ≲ 0.002 is noise.
- **Depth: no collapse to 48L** (+0.041 vs 20L optimum, U-shaped in depth
  at iso-param); H3a's trainability claim holds without gates.
- Figure: `assets/e3_depth.png`.

### E4 — Gap decomposition (H2; interim 2026-07-19 — 31B pair done)
`san eval --checkpoint <ckpt> --by-exercise --by-region` — 20k head-sampled
val docs with metadata, 17.1M scored tokens. 105B-pair heatmap awaits E1.

SAN vs FFN-isoP @31B (E2 finals, both true 30k checkpoints), by region:

| Region | SAN | FFN-isoP | Δ (SAN−FFN) |
|---|---|---|---|
| query | 2.7521 | 2.7004 | **+0.0517** |
| trace | 1.9575 | 1.9500 | +0.0075 |
| answer | 2.0590 | 2.0478 | +0.0112 |
| all | 2.0209 | 2.0096 | +0.0113 |

Answer-region Δ by exercise (+ = SAN worse): memorization +0.011 ·
constrained-writing +0.042 · creative-writing +0.038 · mcq +0.002 ·
math-mcq +0.004 · cooking −0.002 · math **−0.013** · rag **−0.016** ·
editing **−0.017** (noise floor ~0.002).

- **H2 as literally stated is wrong**: the gap does NOT peak on
  memorization-answers — it peaks on QUERY tokens (+0.052, 5× any other
  region), i.e. low-context prediction where parametric knowledge
  substitutes for having something to route.
- **The routing-vs-storage signature appears as a sign flip instead**:
  SAN is BETTER on context-grounded answers (rag/editing/math — the answer
  is retrievable from prompt or trace) and worse on parametric
  (memorization) and free-form generation (creative/constrained writing).
  Trace gap smallest of the substantive regions (+0.008), as predicted.
- MCQ answer regions cost ~0.02 nats for BOTH arms — the answer token is
  fully determined by the trace; traces carry the computation.
- isoF row valid (true 100k ckpt): all-loss 1.7198. **isoD row void** —
  HF held a 40k-step state (upload incident, §10); redo after the tail
  rerun.

### E5 — Data scaling (CLOSED 2026-07-19; 6 × 31.5B = 189B + E2 full-data cells)
`--max-docs` ∈ {2M, 8M, 32M, full}; runs `e5_{arm}_muon_<docs>docs_31B_s42`,
30k steps, 8×H100 512-row frame (matches its E2 comparators). Val @30k:

| Unique docs | Epochs | SAN | FFN-isoP | Δ (SAN−FFN) |
|---|---|---|---|---|
| 2M | ~18× | 2.1193 | **2.0975** | +0.0218 |
| 8M | ~4.5× | 2.1117 | 2.1018 | +0.0099 |
| 32M | ~1.1× | 2.1095 | **2.0926** | +0.0169 |
| 78M (full, E2) | ~0.46× | 2.1126 | 2.0933 | +0.0193 |

- **P8 NOT supported**: the gap shows no monotone trend in repetition
  (0.022 / 0.010 / 0.017 / 0.019) — the FFN does not benefit differentially
  from repeated data at this scale.
- **Both arms are strikingly repetition-insensitive**: 18 epochs over 2M
  docs costs SAN +0.010 and FFN +0.004 vs their best — consistent with
  PleIAs' claims about synthetic-data robustness to repetition.
- Within-arm non-monotonicity (~0.005–0.009) ≈ data-subset variance —
  comparable to the repetition effect itself; single-seed cells, treat
  small Δs accordingly.
- Figure: `assets/e5_data_scaling.png`.

### E6 — Downstream evals (interim 2026-07-19 — 31B pair done)
`san eval --checkpoint <ckpt> --tasks lambada_openai hellaswag arc_easy
piqa sciq winogrande mmlu` (0-shot loglikelihood). 105B rows await E1;
Monad-56M/Baguettotron-321M comparison at write-up.

| Task (0-shot) | chance | SAN 31B | FFN-isoP 31B | FFN-isoF 105B |
|---|---|---|---|---|
| lambada acc | ~0 | 0.091 | 0.136 | **0.178** |
| lambada ppl | — | 820.7 | 492.4 | **451.3** |
| sciq | 0.25 | **0.725** | 0.702 | 0.634 |
| arc_easy | 0.25 | 0.397 | 0.416 | **0.419** |
| piqa | 0.50 | 0.545 | 0.563 | 0.562 |
| hellaswag acc_norm | 0.25 | 0.279 | 0.284 | **0.293** |
| winogrande | 0.50 | 0.504 | 0.516 | **0.534** |
| mmlu | 0.25 | 0.236 | 0.233 | 0.230 |

- Real signal at 24M/31B: sciq, arc_easy, lambada, piqa (weak); hellaswag/
  winogrande/mmlu ≈ chance at this scale — expected, keep for the 105B rows.
- **The E4 sign flip replicates downstream**: lambada — predicting a
  specific content word, pure parametric recall — is FFN's biggest win
  (1.67× ppl ratio vs a 1.1% overall val-loss gap). sciq — the one task
  with a support passage in context — is SAN's only win (+0.023).
  Storage-vs-routing, measured by a third independent instrument.
- isoF (43M, 105B): parametric-recall tasks keep improving with
  params×tokens (lambada 0.178, winogrande first above-chance at 0.534) —
  but **sciq DROPS to 0.634** (−0.07 vs the 24M/31B models, ~4.5σ): more
  capacity/tokens strengthened priors that compete with the in-context
  support passage. Watch whether the 105B SAN row shows the opposite.
- FFN-isoD row VOID (evaluated the 40k-state ckpt — §10 incident); as-run
  values parked for comparison after the rerun: lambada 0.139/632.7,
  sciq 0.653, arc_easy 0.418, piqa 0.568, winogrande 0.501, mmlu 0.256.

### E7 — Mechanistic (interim 2026-07-19; weight spectra over milestones)
Rank + gate trajectories from wandb; `san spectra <ckpts>` over milestone
checkpoints. Eval outputs persist via `--upload-results` → HF `results/`.

Weight stable rank (energy concentration, ‖A‖²_F/‖A‖²₂), SAN base:

| Matrix | init | muon @75% | adamw @75% |
|---|---|---|---|
| q_proj | 129 | **72** | 29 |
| k_proj | 89 | **53** | 30 |
| out_proj | 130 | **86** | 30 |
| v_proj | 89 | 70 | 68 |

- **Muon holds q/k/out spectra 2–3× flatter than AdamW** — the mechanistic
  face of the optimizer story; v_proj is optimizer-invariant (~70 both).
  Same pattern in the FFN cell (adamw attn-q stable rank 17 vs muon 52).
- **Routing freezes, content accumulates** — in every cell: q/k stable
  ranks are set by ~25% of training and never move (SAN 105B q: 75.0 →
  72.8 → 73.1 at 25k/50k/75k), while the write-back/content matrices grow
  monotonically for as long as training runs — FFN down_proj (isoF:
  148 → 156 → 160), attention out_proj (SAN 105B: 90.1 → 94.7 → 94.9;
  FFN-isoP 31B: 93.8 → 101.2), and v_proj (73.4 → 76.4). Remove the FFN
  and the SAN's out_proj inherits the storage-accumulation role — the
  weight-space counterpart of the E4/E6 findings. AdamW's collapse also
  happens in the early window (q stable rank 27.8 by step 7.5k).
- Residual variants (gated/rezero/standard) spectrally indistinguishable,
  matching their equal losses.

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
| ~~lm-eval loglikelihood adapter~~ DONE 2026-07-17 (`san eval --checkpoint X --tasks sciq …`; needs `pip install lm-eval` on the eval pod; validated: sciq acc 0.70 @limit 20 on E2 SAN ckpt) | eval.py |
| ~~SV-spectra script~~ DONE 2026-07-17 (`san spectra <ckpts> --out spectra`; per-kernel SVs + effective/stable rank to npz) | spectra.py |
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
- **2026-07-19 — LADDER + PROBES CLOSED.** Base LR flat/interior at tiny
  and xl; gap plateaus at ~0.02 nats from base up, SAN wins at tiny.
  Tables: §3.
- **2026-07-19 — E5 CLOSED.** P8 not supported; both arms
  repetition-insensitive (18× costs ≤0.010). Table: §4 E5.
- **2026-07-19 — E1 6/8, E3 13/16.** E1: iso-param gap 0.0055 at 105B
  (s42); ffnisop-s43 terminal instability documented. E3: QK-norm removal
  diverges; gates performance-neutral; sandwich −0.009 (only variant
  beating baseline); 48L trains. Tables: §4 E1/E3.
- **2026-07-19 — isoD CHECKPOINT INCIDENT.** All ffnisod HF uploads after
  step 40k silently failed (504s, one-shot uploader); POD 1 terminated →
  100k final + 50k/75k milestones lost. Survivors: step0/25k/40k. Final
  val (1.5982) safe in wandb. Fix: uploader now retries 5× with backoff;
  tail rerun (40k→100k, ~12h) queued for POD 2. E4/E6/E7 isoD rows wait.
- **2026-07-19 — E4 INTERIM (31B pair).** H2 restated: gap peaks on QUERY
  tokens (+0.052), not memorization-answers; sign flip on answers — SAN
  better on context-grounded (rag/editing/math), worse on parametric +
  free-form. Table: §4 E4.
- **2026-07-19 — E6 INTERIM (31B pair).** Downstream replicates the E4
  signature: FFN dominates lambada (parametric recall, 1.67× ppl); SAN
  wins sciq (support passage in context). Table: §4 E6.
- **2026-07-19 — E7 INTERIM.** Muon keeps q/k/out weight spectra 2–3×
  flatter than AdamW (v_proj invariant); spectra freeze by ~25% of
  training; FFN down_proj rank grows throughout — storage accumulation
  visible in weight space. Table: §4 E7.
