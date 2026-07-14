# Simple Attention Networks — Theory

Status: v1, 2026-07-14. Companion to `experiments.md`. Every claim here is
labeled **[Fact]** (derivable or established), **[Cited]** (established
empirically elsewhere), or **[Hypothesis]** (ours; falsifiable; mapped to an
experiment in §7). The README gives the intuition; this document is the
version that has to survive review.

---

## 1. Setup

One SAN block, exactly as implemented (`src/architecture.py`):

```
u   = ZCRMSNorm(x)                        ZCRMSNorm(z) = (1+γ)·z / RMS(z), γ init 0
q,k,v = W_q u, W_k u, W_v u               GQA: n_kv < n_heads
q,k = RoPE(ZCRMSNorm_head(q,k))           per-head QK-norm
A   = softmax(q kᵀ/√d_h + M)              M: causal + document mask
y   = x + σ(g) · W_o (A v)                g: scalar per layer, init 0
```

Stack of N such blocks between tied embeddings; final ZCRMSNorm; softmax LM
head. The FFN control arm inserts `x + σ(g₂)·W_down φ(W_gate u', W_up u')`
after attention (SwiGLU φ).

**The complete inventory of nonlinearities in SAN** [Fact]: (i) the softmax
producing A, (ii) RMS normalization (input and QK), (iii) the output softmax.
The scalar gates are constants w.r.t. the input. There is *no* per-position
learned nonlinear feature map — that is the entire experimental manipulation.

## 2. What attention-only computation is

**Claim 2.1 [Fact]. Attention is an input-dependent linear operator.**
Fix the attention pattern A(x). Then the map from the (normalized) context
to the layer's update is linear: Δᵢ = σ(g)·W_o Σⱼ Aᵢⱼ W_v uⱼ. All
nonlinearity in x enters through A(x) and through normalization. A SAN layer
therefore *selects and linearly transports* content; it does not synthesize
new per-position features nonlinearly.

**Claim 2.2 [Fact]. The degenerate case that defines the bet.** For sequence
length T=1 (or a token that attends only to itself), A = 1 and the entire
network collapses to a chain of {normalize → linear → scaled add}. Up to the
normalizations (which only rescale by a function of the norm), this is a
linear map: a T=1 SAN cannot even represent XOR of two input features.
**Every nontrivial computation in SAN must be cross-token.** This is the
sharpest statement of the architecture's inductive bias, and the honest
statement of its restriction.

**Claim 2.3 [Fact]. Geometry of the update.** Per head, Σⱼ Aᵢⱼ vⱼ is a convex
combination of value vectors — the update to a position lives in the affine
span of (linearly transformed) context content. "Retrieval-and-assembly" is
this statement, made precise: SAN layers move information that already exists
somewhere in context; they cannot conjure representations unsupported by it.

**Claim 2.4 [Cited]. Attention-only stacks are nonetheless expressive across
tokens.** Two attention-only layers implement induction heads (copy/complete
patterns) via K-composition of QK/OV circuits (Elhage et al. 2021, *A
Mathematical Framework for Transformer Circuits*), and composed heads give
rich routing programs at depth. What is missing relative to a transformer is
not sequence computation but pointwise feature synthesis.

**Claim 2.5 [Hypothesis H1]. Language-model pretraining on reasoning traces
is dominated by cross-token computation.** If most of the loss reduction on
trace-style data comes from routing, comparing, and copying visible content
— rather than per-position feature synthesis or parametric recall — then the
restriction in 2.2 is cheap on this distribution, and SAN matches an
iso-parameter transformer. This is the paper's headline claim; it is not
derivable, only measurable (E1).

## 3. Why reasoning traces are the favorable regime

**Claim 3.1 [Cited, with a scope caveat]. Generating intermediate tokens
strictly extends what constant-depth transformers can compute.** Without
chain-of-thought, log-precision constant-depth transformers lie in TC⁰; with
polynomially many generated intermediate steps they can simulate
polynomial-time computation (Merrill & Sabharwal 2023/24; Feng et al. 2023).
Serial compute can be moved from *depth* to *sequence length*. **Caveat:
these constructions use MLPs.** They do not automatically transfer to
attention-only transformers, so 3.1 motivates the data regime but proves
nothing about SAN itself; an attention-only analogue is an open question
(and would be a real theory contribution if established).

**Claim 3.2 [Hypothesis]. SAN benefits differentially from traces.** SAN has
strictly weaker per-layer per-position transformation than the FFN control
(§2.2), but identical ability to consume sequence-length compute (§2.4). If
3.1's mechanism is active in SYNTH's traces, the SAN–FFN gap should be
smaller on trace tokens than on answer tokens (E4 heatmap, region axis).

## 4. Where FFNs should matter: parametric storage

**Claim 4.1 [Cited]. FFN layers act as key–value memories for training
facts.** Geva et al. 2021; causal-editing evidence that factual associations
localize in mid-layer MLPs (ROME, Meng et al. 2022; MEMIT). 

**Claim 4.2 [Fact + honesty]. Attention also stores.** W_qk/W_ov are
associative memories too (Elhage et al. 2021), and fact recall is not
exclusively MLP-localized. So "FFNs store, attention routes" is a *tendency
with causal evidence*, not a theorem. The clean falsifiable form:

**Claim 4.3 [Hypothesis H2].** Removing FFNs at fixed parameters shifts the
loss penalty toward tokens whose prediction requires parametric recall
(memorization-exercise answer regions) and away from tokens predictable by
in-context routing (trace regions, RAG exercises). Measured as the
exercise × region Δloss heatmap (E4). A uniform gap falsifies this.

## 5. Trainability at depth

### 5.1 What the rank-collapse literature actually licenses

Dong et al. 2021 prove pure self-attention networks (no residual, no MLP)
converge doubly-exponentially in depth toward rank-1 token uniformity, and
that **skip connections are the dominant preventer** — with skips, the "path
decomposition" retains identity paths and collapse no longer follows; MLPs
merely slow it further. SAN keeps residuals everywhere. Therefore [Fact]:
**Dong et al. does not predict SAN collapses.** The honest open question is
not token-uniformity collapse but *spectral* degradation of the learned
maps, addressed next. (The README's earlier framing overstated what needed
preventing; this is the corrected statement.)

### 5.2 Signal propagation at initialization [Fact, derivable]

At init: branch input is RMS-normalized (ZCRMSNorm, γ=0 → pure normalize);
W_o uses `residual_init` std 0.02/√(2N); the gate contributes σ(0)=0.5.
Per-layer additive variance is therefore Θ(1/N), and after N layers the
residual-stream variance is Var(x₀) + Θ(1) — **bounded, independent of
depth**. Every component of that sentence is load-bearing: remove the 1/√N
init scaling or the gate damping and stream variance grows linearly in N;
remove normalization and it compounds. This is the standard argument family
of GPT-2's residual scaling, ReZero (Bachlechner et al. 2020), SkipInit
(De & Smith 2020), and LayerScale (Touvron et al. 2021) — our gate is the
scalar, sigmoid-parameterized member of that family, and we claim recipe,
not novelty. What the sigmoid buys over ReZero's raw α: (i) a bounded
multiplier in (0,1) — a layer cannot amplify its own branch arbitrarily;
(ii) init at half strength rather than zero, so every layer receives
gradient from step one. Whether that matters empirically is E3's
residual-variant row, not an assumption.

### 5.3 Zero-centered norm gain [Fact]

Parameterize the norm gain as (1+γ) with γ init 0 and weight decay λ‖γ‖².
Decay then pulls the gain toward **1 (neutral)**. Under the standard
parameterization (gain g, init 1), the same decay pulls the gain toward
**0 (signal annihilation)**. Zero-centering makes "do nothing" the
regularizer's fixed point. That is the entire, sufficient justification.
(Attribution note: this device appears in Gemma-style RMSNorm offsets and
recent stability recipes; nGPT is a related but distinct normalization
program — the README's "from nGPT/DeepSeek-V3" should be corrected.)

### 5.4 Muon: precise claim and honest status

What Muon does [Fact]: Newton–Schulz orthogonalization maps the momentum
gradient to (approximately) its nearest semi-orthogonal matrix — the update
applied to each kernel has all singular values ≈ equal. It does **not** make
the weights orthogonal.

Why that could matter *more* without FFNs [Hypothesis H3-mechanism]: SAN's
per-token pathway is a deep product of linear maps (§2.1). Products amplify
spectral imbalance multiplicatively (κ(∏Wᵢ) can grow exponentially in
depth), and SGD/Adam updates are typically low-effective-rank, feeding that
imbalance. Spectrally flat updates plausibly keep kernel spectra balanced,
and balanced kernels keep deep compositions well-conditioned. In an FFN
transformer, interleaved pointwise nonlinearities partially reset this
dynamic; in SAN nothing does.

Status: this is a mechanism hypothesis, not a theorem. Predicted signatures:
(i) the optimizer × architecture interaction — AdamW should cost SAN more
than it costs the FFN control (E2); (ii) kernel singular-value spectra should
be measurably flatter under Muon, with the gap wider in the SAN arm (E7);
(iii) per-layer effective rank of representations should stay high in
SAN+Muon and degrade in SAN+AdamW if anywhere (E7). If (i) shows no
interaction, Muon is a generic optimizer win and H3 loses its architecture-
specific content.

## 6. Accounting [Fact]

Per layer, parameters: attention (GQA 2:1) = 3d²; SwiGLU FFN at d_ff = 4d =
12d². FFN share of block parameters = 80% in our control config (the classic
"2/3" figure assumes MHA + 2-matrix 4d FFN; both are config-dependent —
state the config when quoting). Training FLOPs per token at T=2048, d=512:
attention ≈ 1.57M (projections) + 4.19M (scores/AV, quadratic in T) per
layer; FFN ≈ 6.29M per layer. Hence the three matchings in `experiments.md`
§3 diverge honestly: iso-param gives SAN ~2× FLOPs/token; iso-FLOP gives the
FFN arm ~2.2× parameters. No single matching is "the fair one"; the paper
reports all three.

## 7. Predictions → experiments

| # | Claim | Prediction | Falsified by | Exp |
|---|---|---|---|---|
| P1 | 2.5 (H1) | SAN ≈ FFN-isoP val loss at 105B tokens | gap > noise band across seeds | E1 |
| P2 | 3.2 | Δloss(SAN−FFN) smaller on trace regions than answer regions | uniform or reversed gap | E4 |
| P3 | 4.3 (H2) | Δloss largest on memorization-answer cells | uniform heatmap | E4 |
| P4 | 5.2 | SAN trains stably to 48 layers with gates; no-residual arm diverges; standard-residual arm degrades at depth | gated ≈ ungated everywhere | E3 |
| P5 | 5.3 | ZCRMSNorm ≥ RMSNorm(γ=1) under identical decay | RMSNorm ≥ ZCRMSNorm | E3 |
| P6 | 5.4 | optimizer×architecture interaction: (AdamW−Muon) penalty larger for SAN than FFN | parallel penalties | E2 |
| P7 | 5.4 | flatter kernel spectra + higher representation rank under Muon, differentially in SAN | spectra indistinguishable | E7 |
| P8 | 4.2 | FFN arm gains more from data repetition (`--max-docs`) | equal repetition curves | E5 |

## 8. Known weaknesses of the argument (state them before reviewers do)

1. **The convex-transport restriction (2.3) is real.** Tasks needing genuine
   per-position feature synthesis exist; we claim they are rare *in this
   distribution*, not absent. The claim is scale- and data-local until the
   ladder (E1) and other corpora say otherwise.
2. **Attention stores too (4.2)** — H2's decomposition could come out
   muddier than the clean story; the heatmap is designed to show the
   tendency's *size*, not to assume it.
3. **Normalization is a per-position nonlinearity.** SAN is not a linear
   model even at T=1 (RMS normalization is nonlinear); 2.2's "up to
   normalization" qualifier is doing work, and we keep it explicit.
4. **Muon's story (5.4) is the most speculative layer** of the stack and is
   deliberately fenced as mechanism-hypothesis + measurement.
5. **Efficiency asymmetry (§6):** at iso-param SAN buys its parameter
   savings with attention FLOPs that are quadratic in context length. At
   T=2048 this is a good trade on modern accelerators; at long context it
   degrades. We scope claims to the measured regime.

## 9. Formal results (appendix candidates — statements with proofs)

Assumptions are explicit. Notation:
one block at position i computes yᵢ = xᵢ + σ(g)·W_o Σⱼ Aᵢⱼ(x) W_v ûⱼ, with
û = diag(1+γ)·x/ρ(x), ρ(x) = RMS(x), A row-stochastic (softmax).

**Proposition A (conditional linearity).** Condition on the attention
pattern: for any fixed row-stochastic A, the map (û₁,…,û_T) ↦ (Δ₁,…,Δ_T)
is linear. *Proof.* It is the composition of the linear maps W_v, v ↦ Av,
and W_o, scaled by the constant σ(g). ∎

**Proposition B (simplex transport).** Per head, the pre-projection update
Σⱼ Aᵢⱼ (W_v ûⱼ) lies in the convex hull of {W_v ûⱼ : j ≤ i}. *Proof.*
Softmax rows are nonnegative and sum to 1. ∎
(Interpretation: a head transports content present in context; it does not
leave the convex hull of transformed context vectors.)

**Lemma C (scalar-bottleneck structure at T=1).** For sequence length 1,
A = [1] and RoPE at position 0 is the identity, so each block reduces to
x_{ℓ+1} = (I + B̃_ℓ / ρ_ℓ) x_ℓ, where B̃_ℓ = σ(g_ℓ)·W_o W_v diag(1+γ_ℓ) is
constant and ρ_ℓ = RMS(x_ℓ). By induction,
x_L = [ ∏_ℓ (I + B̃_ℓ/ρ_ℓ) ] x₀ :
the entire depth-L network is a linear map modulated by exactly L scalars.
*Corollaries.* (i) Restricted to any sphere ‖x₀‖ = r, the first block is
linear. (ii) Any two inputs with identical radius trajectories (ρ₁,…,ρ_L)
are processed by the *same* linear map. All per-position nonlinearity in
SAN flows through this L-scalar bottleneck; cross-token attention is the
only escape. ∎

**Theorem D (bounded stream variance at init).** Assume the initialization
as implemented: γ = 0 (norms are exact RMS-normalizations), gates σ(0) = ½,
W_o entries i.i.d. mean-zero with variance s²/(2N), W_v fixed-scale, all
independent. Then for every depth N,
E‖x_N‖² = E‖x₀‖² + Σ_ℓ E‖b_ℓ‖², with E‖b_ℓ‖² = Θ(1/N),
hence E‖x_N‖² = E‖x₀‖² + Θ(1) uniformly in N.
*Proof sketch (complete in appendix).* Cross terms E⟨x_ℓ, b_ℓ⟩ vanish
because W_o is mean-zero and independent of everything upstream. Each
branch input is RMS-normalized, so E‖z_ℓ‖² is bounded independent of the
stream magnitude (this is what pre-norm buys); row-stochastic A cannot
increase the max value norm (Prop. B); the 1/(2N) output variance then
gives E‖b_ℓ‖² = Θ(1/N), and the sum telescopes. ∎
(Each hypothesis is load-bearing: drop the 1/√(2N) scaling or the gate and
the sum is Θ(1) per layer → Θ(N) total; drop pre-norm and growth compounds.)

**Proposition E (weight-decay fixed point).** Under L2 decay, the penalty
on γ is minimized at γ = 0, i.e. norm gain 1 (identity). Under the standard
parameterization (gain h, init 1), the penalty is minimized at h = 0, the
zero map. Zero-centering makes "do nothing" the regularizer's unique fixed
point. *Proof.* Both objectives are strictly convex with the stated
minimizers. ∎

**Lemma F (Muon bounds per-step spectral drift — and only that).** Let the
update be Δ = −η·P with P the polar factor of the momentum gradient (all
singular values of P equal 1; Newton–Schulz approximates this to error ε).
By Weyl's inequality for singular values, for every i:
|σᵢ(W + Δ) − σᵢ(W)| ≤ ‖Δ‖₂ = η(1+ε).
So under Muon *every* singular value of every kernel drifts at most η per
step — after t steps, spectra lie in [σᵢ(0) − tη, σᵢ(0) + tη]. A general
(Adam-style) update with the same Frobenius budget can concentrate up to
√rank × more of its energy on the top singular direction per step.
*Scope.* This bounds drift; it does **not** prove spectra stay balanced or
that balance improves loss — that remainder is exactly what E7 measures. ∎

**What has no theorem:** H1 and H2 (empirical,
by design); the claim that Muon's drift bound translates into better
conditioning of the *learned product* over full training (open; E2/E7);
any CoT-expressivity claim for attention-only models (open — see 3.1
caveat).
