# Simple Attention Networks

We show that MLPs can be completely dropped from transformer networks.

```
d=512, 8H/4KV, BPE=8192
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
      ┌───────┴────────┐
      │  Block x 20    │
      │ ┌────────────┐ │
      │ │ ZCRMSNorm  │ │
      │ │ Masked Self│ │
      │ │ Attn       │ │
      │ │ GQA + RoPE │ │
      │ │ Gated Res  │ │
      │ │            │ │
      │ │  (no FFN)  │ │
      │ └────────────┘ │
      └───────┬────────┘
      ┌───────┴───────┐
      │   Embedding   │  ← shared
      └───────┬───────┘
      ┌───────┴───────┐
      │  Text tokens  │
      └───────────────┘
```

Decoder-only, pretrained on streaming HF datasets (default [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)).
The FFN-equipped standard transformer is the built-in control arm (`--ffn`), and the
optimizer is switchable (`--optimizer muon|adamw`), so the {muon, adamw} × {ffn, no-ffn}
2×2 runs from one code path. Experiment design: [docs/neurips_experiment_plan.md](docs/neurips_experiment_plan.md).

## Why No FFN

- **Softmax is nonlinear.** `softmax(QK^T/sqrt(d)) * V` is a data-dependent nonlinear mixing operation. For a task that is about routing information (query -> tool alignment), attention is the right primitive.
- **Tool calling is retrieval-and-assembly.** Match query to tool name, extract argument values, assemble JSON. All three are aligning and copying between input and output -exactly what cross-attention does. No step requires per-position feature transformation (which is what FFN provides).
- **At small scale, FFN parameters are wasted.** ~2/3 of standard transformer parameters are FFN. For a <50M model on a structured task, those parameters contribute less than more attention layers (deeper cross-attention = better query-tool alignment).
- **Fewer parameters = faster inference.** FFNs have the biggest GEMM/GEMV dimensions -removing them cuts per-layer parameters by ~2/3, directly reducing the memory bandwidth bottleneck that dominates latency on edge devices.

## Gated Residuals

Without FFN, there is no per-position nonlinear rewriting per layer. This makes residual connection design critical.

- **Standard residual** `x = x + Attn(Norm(x))` -attention can only ADD a delta. Without FFN to do the rewriting, purely additive is limiting.
- **No residual** `x = Attn(Norm(x))` -each layer fully rewrites, but we lose the gradient highway. Deep networks (12+ layers) will not train.
- **Gated residual (ours)** `x = x + sigmoid(gate) * Attn(Norm(x))` -per-sublayer learnable scalar, initialized to 0. sigmoid(0) = 0.5, so training starts with half-strength residual. The model can learn to sharpen useful layers (g->1) or suppress unhelpful ones (g->0) without losing gradient flow.

## ZCRMSNorm

- **Standard RMSNorm:** `x * gamma / RMS(x)`, gamma initialized to 1.
- **ZCRMSNorm:** `x * (1 + gamma) / RMS(x)`, gamma initialized to 0.
- At init, ZCRMSNorm is identity-up-to-scale. Pairs with gated residuals: the entire block starts as a damped identity + damped normalized attention. No component starts with a strong learned bias.
- From the nGPT / DeepSeek-V3 line of work. Applied to QK heads as well (QK-norm) for training stability.

## Muon for Attention-Only

- **Dual optimizer:** Muon (Q/K/V/O projections, LR 0.02, WD 0.01) + AdamW (everything else, LR 3e-4).
- Without FFN, the model is a deep stack of linear projections with softmax routing. 
- Muon enforces orthogonality on weight updates via Newton-Schulz, preventing the representation collapse that can happen when stacking many linear layers without interleaving nonlinearities.

## Quickstart

```bash
git clone https://github.com/cactus-compute/needle.git
cd needle && source ./setup

# 1. Train the tokenizer on the pretraining corpus (once; --upload for TPU pods)
needle tokenizer-train

# 2. Pretrain the SAN arm
needle pretrain --wandb

# 3. Control arms
needle pretrain --ffn --wandb                       # + FFN
needle pretrain --optimizer adamw --wandb           # - Muon
needle pretrain --ffn --optimizer adamw --wandb     # standard transformer
```

Swap datasets with `--dataset fineweb-edu` or any HF repo id + `--text-field`.
Documents are packed into fixed-length rows with segment IDs; attention and
the loss are masked at document boundaries.

## CLI

```
needle pretrain          Streaming pretraining
needle eval              Val loss/PPL + throughput + sample generations
needle sample            Prompt continuation from a checkpoint
needle tokenizer-train   Train the SentencePiece tokenizer (vocab 8192)
needle tpu <action>      TPU management (see docs/tpu.md)
```

Key pretrain flags: `--ffn/--no-ffn`, `--optimizer {muon,adamw}`, `--dataset`,
`--seq-len`, `--batch-size` (per device), `--max-steps` (also the WSD schedule
horizon), `--num-layers`, `--d-model`, `--seed`, `--eval-every` (val loss +
per-layer gate logging), `--log-rank-every` (per-layer representation rank via
SVD entropy), `--upload-checkpoints`.

## Conventions

- **LR scaling across devices:** Adam LR scales linearly (`--lr × total_devices`);
  Muon LR scales by sqrt (`--muon-lr × sqrt(total_devices)`). Both optimizer arms
  inherit the same convention.
- **Weight decay** applies to Dense kernels only, identically in both optimizer
  arms (0.01).
- **Checkpoints** are format v2 (`{format_version, params, config, step, run}`),
  named `<--name>.pkl`. Old encoder-decoder/tool-calling checkpoints and the old
  `needle.model` tokenizer are incompatible with this branch by design.
- **Batch size** is per device: global batch = `--batch-size × devices × hosts`.

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
  url={https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md}
}
```
