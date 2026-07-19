# Paper Structure Notes — "A Controlled Study of Attention-Only Transformers"

AAAI-27, 7 pages two-column + supplementary. Distilled from four review
rounds (6 → 7 → 8): every point gained came from surfacing controls we
already had, never from new experiments. Write accordingly.

## Principles (from the review back-and-forth)

1. **Every claim arrives with its defense in the same sentence.** Never let
   a number stand alone; the reviewer's first move is always "vs what?"
2. **Known attack order** (observed, in sequence): effect size vs noise →
   "this is known" novelty → framing tension (delete vs reallocate) →
   corpus selection → scale ceiling → dropped threads (all 3 matchings must
   appear) → reflex numeric checks (every pair of numbers must reconcile or
   carry scope labels: non-emb vs total, which token budget, which seed).
3. **Own indistinguishability as the result.** "Statistical parity at
   matched parameters (Δ = x ± σ, n=3, paired bootstrap)" — never defend
   0.006 as a precise effect size.
4. **Limitations become predictions; predictions become measurements when
   cheap.** The fineweb pre-registration (0.02–0.05 window, dated before
   the run) is the template. State the scale boundary as the storage
   account's own prediction.
5. **The monotone sequence is the headline**: 0.47 (delete in place) →
   0.26 (iso-FLOP) → 0.006 (reallocate). The ordering itself is evidence —
   the gap tracks reclaimable parameter budget. Lead with it everywhere.
6. **Report the unfavorable number prominently** (iso-FLOP 0.26, attention
   pays the quadratic term). Credibility for the favorable ones.

## Section plan (page budget)

1. **Introduction (1.0)** — the untested default (2/3 of non-emb params);
   the three-matching necessity test as the methodological gap (scoped
   novelty claim, NOT "never tested"); the monotone sequence with seed σ
   in the first half page; contributions list ranked: (i) controlled
   test + parity result, (ii) storage/routing decomposition ×3
   instruments, (iii) QK-norm necessity, (iv) spectra dynamics
   (routing freezes / content accumulates / relocation), (v) recipe facts.
2. **Related work (0.5)** — from literature.md. Credit FFN-as-memory
   lineage (Geva, ROME, Dai) BEFORE stating our conclusion so it reads as
   confirmation-by-ablation, not rediscovery. Distinguish Sukhbaatar
   (compensated removal) in one sentence. Dong et al. as the theory this
   updates. gMLP/Mixer as the dual direction.
3. **Setup (0.75)** — SAN block diagram; matchings table (all four
   configs); fixed frame; per-arm LR sweep protocol (E0, boundary rule);
   **noise-floor calibration** (rmsnorm optimizer-equivalence control,
   same-seed Δ = 0.0015) — spend space here, it licenses every later
   "within noise" claim.
4. **Main result (1.25)** — three-control sequence; token axis (separately
   trained 5B/30B/105B); parameter axis (ladder: flat ~0.02 base→xl,
   reversal at tiny with the depth-floor explanation); seed CIs + paired
   bootstrap. Fig 1 (val curves), Fig 2 (ladder). One paragraph:
   repetition insensitivity (E5, P8 null) — or supplementary if tight.
5. **Where the gap lives (1.25)** — the paper's discovery section. Region
   × exercise heatmap; additive decomposition closure (Σ shares × Δ =
   aggregate, verified); loss shares (query 8% of loss, 5× per-token Δ);
   scale-resolved region gaps (ladder E4); downstream signature (lambada
   vs sciq, isoF sciq regression); fineweb result vs the pre-registered
   window. Fig 3 (heatmap or Δ-by-region bars across sizes).
6. **Mechanism (0.75)** — weight spectra: routing (Q/K) freezes by 25%,
   content (V/out, down_proj) accumulates rank throughout; removal
   relocates accumulation to out_proj; Muon holds routing spectra 2–3×
   flatter than AdamW (v_proj invariant). Fig 4 (stable-rank
   trajectories, muon vs adamw overlay).
7. **Component ablations (0.5)** — E3 table: QK-norm removal diverges
   (scope: at the recipe's tuned LR — say this, or a reviewer asks whether
   a lower LR survives without it); sandwich −0.009 (only improvement,
   adopted in final recipe with the studied-vs-recommended distinction);
   gates performance-neutral but diagnostic; depth U-curve to 48L.
8. **Discussion + limitations (0.5)** — scope box: ≤87M params, one
   reasoning-dense corpus, ≤105B tokens; the account's own boundary
   predictions (wider gap at scale on storage-heavy mixtures); when
   attention-only is the right trade (param-bound, not FLOP-bound;
   simplicity; analysis value) — be honest that iso-FLOP favors FFN at
   2048 context; planned ~300M pair (rebuttal window, Sept).

## Supplementary (due +3 days, use it hard)

E0 full sweep + locks; E2 2×2 + P6 budget-dependence; E5 tables; LR
probes; instability incidents (ffnisop-s43 tail figure, E0 gate collapse)
framed as FFN-arm fragility observations, not claims; full E6 tables (mmlu
at chance — say why kept); per-layer spectra panels; theory.md results
with the MLP caveat on CoT expressivity theorems; reproducibility: every
run name, seeds, LR locks, frame-matching math for 7-GPU pods, HF
artifact map (checkpoints/, results/, corpus/), commands.

## Claim → defense map (check each pair survives editing)

| Claim | In-paper defense |
|---|---|
| parity at iso-param | 3 seeds + σ + noise floor + paired bootstrap |
| methodological novelty | matched-3-ways scoping + Sukhbaatar distinction |
| gap shrinks with tokens | three separately trained + separately tuned budgets |
| gap flat in params | ladder, labeled non-emb range, fixed 31.5B tokens |
| gap is concentrated | decomposition closure + loss-share matrix |
| region gap grows w/ capacity | ladder E4 (scale-resolved), else cut the claim |
| corpus generality | pre-registered fineweb window + measured result |
| QK-norm necessity | divergence cell, scoped to tuned LR |
| routing/content split | per-layer spectra, multiple sizes + both optimizers |
| storage relocation to W_o | SAN vs FFN trajectories at matched budget |

## Numbers hygiene

- Abstract numbers must match body tables exactly; recompute after s44 /
  isoD / fineweb land: 0.006 (→ mean ± σ), 0.47, 0.26, shares, window.
- Label every parameter count total vs non-embedding at first use.
- Every "within noise" → cite the 0.0015 floor or the seed σ, whichever
  applies.
- The s43 final stays in the mean with the incident note; no silent
  exclusion anywhere.
