# Simple Attention Networks

Can MLPs be dropped from transformer networks entirely? SAN is the
attention-only arm of a controlled test of that question on reasoning-oriented
pretraining. Rigorous argument: [theory.md](theory.md). Experiment design:
[experiments.md](experiments.md).

```
      ┌───────────────┐
      │  Next Token   │
      └───────┬───────┘
         ┌────┴────┐
         │ Softmax │
         └────┬────┘
      ┌───────┴───────┐
      │  Linear (T)   │  ← tied
      └───────┬───────┘
      ┌───────┴───────┐
      │   ZCRMSNorm   │
      └───────┬───────┘
      ┌───────┴─────────────────────────────┐
      │       │                             │   Block × N
      │       ▲ y                           │
      │      (+)◄─────────────────────┐     │   y = x + σ(g)·o
      │       ▲                       │     │
      │     × σ(g)                    │     │   σ(g): learnable scalar gate,
      │       ▲                       │     │   g init 0 → σ(g) = 0.5
      │   o = W_o·(A·v)               │     │
      │       ▲                       │     │
      │   A = softmax(q·kᵀ/√d + M)    │     │   GQA + causal mask
      │       ▲                       │     │
      │   q,k = RoPE(q,k)             │ x   │
      │       ▲                       │     │
      │   q,k = ZCRMSNorm(q,k)        │     │   (QK-norm)
      │       ▲                       │     │
      │   q,k,v = W_q·u, W_k·u, W_v·u │     │   (GQA)
      │       ▲                       │     │
      │   u = ZCRMSNorm(x)            │     │   ZCRMSNorm(z) = (1+γ)·z/RMS(z)
      │       ▲                       │     │   RMS(z) = √mean(z²), γ init 0
      │       ├───────────────────────┘     │
      │       │ x                           │
      └───────┬─────────────────────────────┘
      ┌───────┴───────┐
      │   Embedding   │  ← shared
      └───────┬───────┘
      ┌───────┴───────┐
      │ Input tokens  │
      └───────────────┘
```

Decoder-only, pretrained on streaming HF datasets (default [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)).
The FFN-equipped standard transformer is the built-in control arm (`--ffn`), and the
optimizer is switchable (`--optimizer muon|adamw`), so the {muon, adamw} × {ffn, no-ffn}
2×2 runs from one code path.

## Why No FFN

- **Attention is a data-dependent linear operator.** `softmax(QKᵀ/√d)·V`
  nonlinearly *selects* content, then linearly transports it. A SAN layer can
  route, compare, and copy anything already in context — it cannot synthesize
  new per-position features. Every nontrivial computation is cross-token
  (theory.md §2).
- **Reasoning traces externalize computation into context.** Chain-of-thought
  moves serial compute from depth to sequence length, and attention consumes
  sequence-length compute at full strength. Trace-formatted pretraining data
  (SYNTH) is the regime where the missing FFN should matter least (theory.md §3).
- **FFNs are the parameter budget.** In our control config (GQA + SwiGLU,
  d_ff = 4d), FFNs are ~80% of block parameters. If FFNs mainly store
  parametric knowledge (Geva et al. 2021) and the task's knowledge is in
  context, those parameters buy depth instead: 20 attention-only layers cost
  the same as 4 standard blocks (theory.md §4, §6).

## Gated Residuals

Without FFN, there is no per-position nonlinear rewriting per layer. This makes residual connection design critical.

- **No residual** `x = Attn(Norm(x))` — each layer fully rewrites, but the
  gradient highway is gone; deep stacks will not train.
- **Standard residual** `x = x + Attn(Norm(x))` — trains, but branch
  magnitudes are uncontrolled and stream variance grows with depth.
- **Gated residual (ours)** `x = x + sigmoid(g) * Attn(Norm(x))` — scalar per
  sublayer, g init 0 → half-strength start. Combined with the 1/√(2N) output
  init, residual-stream variance stays bounded independent of depth
  (theory.md §5.2). Layers can sharpen (g→∞) or self-prune (g→−∞) without
  losing gradient flow. Scalar member of the ReZero/SkipInit/LayerScale
  family — recipe, not novelty.

## ZCRMSNorm

- **Standard RMSNorm:** `x * gamma / RMS(x)`, gamma initialized to 1.
- **ZCRMSNorm:** `x * (1 + gamma) / RMS(x)`, gamma initialized to 0.
- At init, ZCRMSNorm is a pure RMS normalize (gain exactly 1). Pairs with
  gated residuals: the whole block starts as a damped identity + damped
  normalized attention.
- The real reason for zero-centering: weight decay on γ pulls the gain toward
  **1 (neutral)**; with the standard parameterization it pulls the gain toward
  **0 (signal annihilation)**. Zero-centering makes "do nothing" the
  regularizer's fixed point (theory.md §5.3). Gemma-style RMSNorm offset;
  also applied per-head to Q/K (QK-norm) to bound attention logits.

## Muon for Attention-Only

- **Dual optimizer:** Muon (Q/K/V/O projections, LR 0.02, WD 0.01) + AdamW (everything else, LR 3e-4).
- Muon orthogonalizes weight **updates** (not weights) via Newton–Schulz:
  every update has a flat singular spectrum.
- The hypothesis (tested, not assumed): without FFNs the per-token pathway is
  a deep product of linear maps, where spectrally imbalanced, low-rank updates
  compound multiplicatively; flat updates keep the composition
  well-conditioned. Predicted signature: AdamW should hurt the SAN arm more
  than the FFN arm (theory.md §5.4, predictions P6–P7).

## Quickstart

```bash
git clone https://github.com/cactus-compute/needle.git
cd needle && git checkout neurips && source ./setup

# 1. Train the tokenizer on the pretraining corpus (once; --upload to share via HF)
san tokenizer-train

# 2. Pretrain the SAN arm
san pretrain --wandb

# 3. Control arms
san pretrain --ffn --wandb                       # + FFN
san pretrain --optimizer adamw --wandb           # - Muon
san pretrain --ffn --optimizer adamw --wandb     # standard transformer
```

Swap datasets with `--dataset fineweb-edu` or any HF repo id + `--text-field`.
Documents are packed into fixed-length rows with segment IDs; attention and
the loss are masked at document boundaries.

## CLI

```
san pretrain          Streaming pretraining
san eval              Val loss/PPL + throughput + sample generations
san sample            Prompt continuation from a checkpoint
san tokenizer-train   Train the SentencePiece tokenizer (vocab 16384)
```

Training is single-node data-parallel via pmap over all local GPUs (e.g. one
RunPod node with 8x H100) — no launcher needed, just `san pretrain`.

Key pretrain flags: `--ffn/--no-ffn`, `--optimizer {muon,adamw}`, `--dataset`,
`--seq-len`, `--batch-size` (per device), `--max-steps` (also the WSD schedule
horizon), `--num-layers`, `--d-model`, `--seed`, `--eval-every` (val loss +
per-layer gate logging), `--log-rank-every` (per-layer representation rank via
SVD entropy), `--upload-checkpoints`.

## Conventions

- **LR scaling across devices:** Adam LR scales linearly (`--lr × num_gpus`);
  Muon LR scales by sqrt (`--muon-lr × sqrt(num_gpus)`). Both optimizer arms
  inherit the same convention.
- **Weight decay** applies to Dense kernels only, identically in both optimizer
  arms (0.01).
- **Checkpoints** are format v2 (`{format_version, params, config, step, run}`),
  named `<--name>.pkl`. Old encoder-decoder/tool-calling checkpoints and the old
  `needle.model` tokenizer are incompatible with this branch by design.
- **Batch size** is per device: global batch = `--batch-size × local devices`.

## Tests

```bash
python -m pytest tests/
```

Covers causal masking (no future leakage), packed-vs-separate forward
equivalence, segment/loss-mask construction, gate parameter paths, and one
train step per {muon, adamw} × {ffn, no-ffn} cell.

## Citation

```
@misc{ndubuaku2026SAN,
  title={Simple Attention Networks},
  author={Henry Ndubuaku},
  year={2026},
  url={https://github.com/cactus-compute/needle}
}
```
