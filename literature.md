# Related Work — "A Controlled Study of Attention-Only Transformers"

Grouped by the paper section each feeds. One entry per work: link, what it
showed, and the relation to this study.

## 1. Attention-only transformers & the role of the FFN

- **Attention Is All You Need** (Vaswani et al., NeurIPS 2017) —
  https://arxiv.org/abs/1706.03762
  Introduces the transformer with the FFN as an unexamined default (2/3 of
  params). Ours: the controlled removal study that default never received.
- **Augmenting Self-attention with Persistent Memory** (Sukhbaatar et al.,
  2019) — https://arxiv.org/abs/1907.01470
  Removes the FFN by folding it into attention as learned "persistent"
  KV vectors; matches standard transformers on LM benchmarks. Closest
  precursor. Ours: removes the FFN *without* adding persistent capacity,
  and decomposes what is lost (storage) vs retained (routing).
- **Attention Is Not All You Need: Pure Attention Loses Rank Doubly
  Exponentially with Depth** (Dong, Cordonnier, Loukas, ICML 2021) —
  https://arxiv.org/abs/2103.03404
  Proves attention-only networks *without residuals* collapse to rank-1;
  residuals + FFN slow this. Ours: E3 residual axis + rank diagnostics test
  this directly — with modern norms/init, 48L attention-only trains (loss
  2.18 vs 2.88 for no-residual), and QK-norm, not the FFN, is what prevents
  divergence.
- **A Mathematical Framework for Transformer Circuits** (Elhage et al.,
  Anthropic 2021) — https://transformer-circuits.pub/2021/framework/index.html
  Studies 1–2 layer attention-only models as an interpretable model class
  (QK = routing, OV = content transport). Ours: trains that class at
  realistic depth/budget and measures the cost; the QK/OV split reappears
  empirically in our spectra (q/k freeze, v/out accumulate).
- **In-context Learning and Induction Heads** (Olsson et al., Anthropic
  2022) — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
  Attention heads implement context copying/retrieval. Ours: explains why
  SANs win precisely on context-grounded regions/tasks (rag answers, sciq).
- **Transformer Feed-Forward Layers Are Key-Value Memories** (Geva et al.,
  EMNLP 2021) — https://arxiv.org/abs/2012.14913
  FFN as pattern-keyed memory over the vocabulary. Ours: the causal
  counterpart — remove the memory and the deficit localizes to parametric
  recall (E4 memorization/query regions, E6 lambada).
- **Transformer Feed-Forward Layers Build Predictions by Promoting
  Concepts in the Vocabulary Space** (Geva et al., EMNLP 2022) —
  https://arxiv.org/abs/2203.14680
  Mechanism refinement of the above. Ours: same relation.
- **Locating and Editing Factual Associations in GPT (ROME)** (Meng et
  al., NeurIPS 2022) — https://arxiv.org/abs/2202.05262
  Factual associations live in mid-layer FFNs. Ours: convergent evidence
  from ablation rather than editing.
- **Knowledge Neurons in Pretrained Transformers** (Dai et al., ACL 2022)
  — https://arxiv.org/abs/2104.08696
  Individual FFN units correlate with facts. Ours: same relation.
- **One Wide Feedforward Is All You Need** (Pires et al., 2023) —
  https://arxiv.org/abs/2309.01826
  FFNs are redundant across layers (share/remove most of them in MT).
  Ours: total removal in decoder-only pretraining, iso-param controlled.
- **Simplifying Transformer Blocks** (He & Hofmann, ICLR 2024) —
  https://arxiv.org/abs/2311.01906
  Removes skips, value/projection params via signal-propagation arguments.
  Ours: same spirit, orthogonal axis (FFN), plus capability decomposition.

## 2. The dual direction (attention-free) and mixer framing

- **MLP-Mixer** (Tolstikhin et al., NeurIPS 2021) —
  https://arxiv.org/abs/2105.01601
  All-MLP vision model — the opposite ablation. Ours: the two ablations
  bracket the transformer; we supply the missing attention-only side for LM.
- **Pay Attention to MLPs (gMLP)** (Liu et al., NeurIPS 2021) —
  https://arxiv.org/abs/2105.08050
  Gated MLPs match transformers in vision, lag on language tasks needing
  cross-token retrieval. Ours: mirror-image finding — attention-only lags
  exactly on tasks needing storage.
- **MetaFormer Is Actually What You Need** (Yu et al., CVPR 2022) —
  https://arxiv.org/abs/2111.11418
  Claims the block frame, not the mixer, drives performance. Ours: a limit
  test of that claim — deleting the channel-MLP entirely costs ~0.006 nats
  at iso-param on our distribution, with a specific residual profile.

## 3. Residuals, gates, normalization (E3 axes)

- **Highway Networks** (Srivastava et al., 2015) —
  https://arxiv.org/abs/1505.00387
  Gated residuals predating transformers. Ours: our scalar sigmoid gates
  are the minimal Highway variant; found performance-neutral but
  diagnostically valuable.
- **ReZero Is All You Need** (Bachlechner et al., 2020) —
  https://arxiv.org/abs/2003.04887
  α-init-0 residual scaling for deep-net trainability. Ours: E3 confirms
  gated ≈ rezero ≈ standard at 20L under modern norms (Δ ≤ 0.003).
- **DeepNet: Scaling Transformers to 1,000 Layers** (Wang et al., 2022) —
  https://arxiv.org/abs/2203.00555
  Residual scaling for extreme depth. Ours: context for the depth ladder;
  we reach 48L attention-only with norms/init alone.
- **Query-Key Normalization for Transformers** (Henry et al., Findings
  EMNLP 2020) — https://arxiv.org/abs/2010.04245
  Introduces QK-norm. Ours: E3's strongest single finding — removing
  QK-norm diverges outright at the tuned Muon LR in attention-only stacks.
- **Scaling Vision Transformers to 22B** (Dehghani et al., 2023) —
  https://arxiv.org/abs/2302.05442
  QK-norm as the fix for attention-logit divergence at scale. Ours: same
  mechanism, load-bearing even at 24M when the stack is attention-only.
- **NormFormer** (Shleifer et al., 2021) — https://arxiv.org/abs/2110.09456
  Extra normalization after attention (sandwich-style). Ours: sandwich is
  our only variant beating baseline (−0.009 nats).
- **CogView** (Ding et al., NeurIPS 2021) — https://arxiv.org/abs/2105.13290
  Sandwich-LN for stability. Ours: same relation.
- **Signal Propagation in Transformers** (Noci et al., NeurIPS 2022) —
  https://arxiv.org/abs/2206.03126
  Rank collapse and LN interaction at init. Ours: theoretical backdrop for
  the init-variance boundedness result and rank diagnostics.
- **2 OLMo 2 Furious** (OLMo team, AI2 2024) — https://arxiv.org/abs/2501.00656
  Production recipe using zero-centered weight decay on norms. Ours:
  ZCRMSNorm choice; we additionally show it is optimizer-equivalent to
  standard RMSNorm under kernel-only WD (measured Δ = 0.0015 = noise floor).
- **Gated Attention for Large Language Models** (Qiu et al., 2025) —
  https://arxiv.org/abs/2505.06708
  Sigmoid output gates on attention improve LLMs. Ours: related gating
  site; our per-sublayer scalar gates are neutral for loss but expose
  self-pruning dynamics (E0 FFN-gate collapse, E2 optimizer split).
- **Attention Residuals** (Kimi Team, Moonshot AI, 2026) —
  https://arxiv.org/abs/2603.15031 · https://github.com/MoonshotAI/Attention-Residuals
  Replaces additive residuals with softmax attention over depth; frames all
  residual variants as depth-mixing matrices (linear attention over depth).
  Ours: E3's residual axis lives in that taxonomy; our gate trajectories
  are small-scale evidence that learned non-uniform depth mixing is useful.

## 4. Optimizer (Muon)

- **Muon** (Jordan et al., 2024) — https://kellerjordan.github.io/posts/muon/
  Orthogonalized momentum for hidden-layer matrices. Ours: primary
  optimizer; E7 shows its mechanism at work (2–3× flatter q/k/out spectra
  vs AdamW).
- **Muon Is Scalable for LLM Training** (Liu et al., Moonshot/Kimi 2025) —
  https://arxiv.org/abs/2502.16982
  Muon at LLM scale with weight decay + update-scale corrections. Ours:
  supports optimizer choice; P6 finds the Muon-vs-AdamW gap is
  budget-dependent (crossover at 5B, gone by 30B).
- **Kimi K2** (Moonshot AI, 2025) — https://github.com/MoonshotAI/Kimi-K2
  MuonClip at 1T-param scale. Ours: external validity for Muon; K3 (2026)
  extends to per-head Muon — optimizer×architecture interaction is a live
  frontier question (H3b).

## 5. Data, scale, and training regime

- **Training Compute-Optimal LLMs (Chinchilla)** (Hoffmann et al., 2022) —
  https://arxiv.org/abs/2203.15556
  Compute-optimal token/param trade. Ours: deliberately over-trained
  (105B tokens at 24M params) to reach the capacity-limited regime H1/H2
  require.
- **Scaling Data-Constrained Language Models** (Muennighoff et al.,
  NeurIPS 2023) — https://arxiv.org/abs/2305.16264
  Repetition up to ~4 epochs nearly free on web data. Ours: E5 extends to
  18 epochs on synthetic data at ≤0.010 nats cost, both arms; no
  architecture×repetition interaction (P8 null).
- **TinyStories** (Eldan & Li, 2023) — https://arxiv.org/abs/2305.07759
  Synthetic data makes tiny LMs coherent. Ours: lineage for small-model
  synthetic pretraining.
- **Textbooks Are All You Need (phi-1)** (Gunasekar et al., 2023) —
  https://arxiv.org/abs/2306.11644
  Curated/synthetic data beats scale. Ours: same lineage.
- **SYNTH / Monad / Baguettotron** (PleIAs, 2025) —
  https://huggingface.co/datasets/PleIAs/SYNTH
  78M-doc reasoning-first synthetic corpus + reference models trained on
  it. Ours: the training distribution; Monad-56M/Baguettotron-321M are the
  external comparison points for E6; our repetition-insensitivity result
  corroborates their claims. Caveat carried in the paper: reasoning-dense
  synthetic data is a routing-friendly distribution — H1's scope is
  explicitly conditioned on it.

## 6. Expressivity theory used in theory.md

- **The Expressive Power of Transformers with Chain of Thought** (Merrill
  & Sabharwal, ICLR 2024) — https://arxiv.org/abs/2310.07923
  CoT steps extend transformer expressivity. Ours: cited for the
  trace-as-computation framing; their constructions use MLPs — theory.md
  carries this caveat explicitly.
- **Towards Revealing the Mystery behind Chain of Thought** (Feng et al.,
  NeurIPS 2023) — https://arxiv.org/abs/2305.15408
  CoT enables serial computation beyond fixed-depth limits. Ours: same
  relation; empirically, MCQ answers cost ~0.02 nats given the trace —
  traces demonstrably carry the computation in our models.

## 7. Evaluation

- **lm-evaluation-harness** (Gao et al., EleutherAI) —
  https://github.com/EleutherAI/lm-evaluation-harness
  Standard 0-shot loglikelihood protocol. Ours: E6 adapter targets it;
  tasks chosen to match the PleIAs evaluation suite.
