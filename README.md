# Simple Attention Networks

We show that MLPs can be completely dropped from transformer networks.

```
d=512, 8H/4KV, BPE=8192
                                  ┌──────────────┐
                                  │  Tool Call   │
                                  └──────┬───────┘
                                        ┌┴──────────┐
                                        │  Softmax  │
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ Linear (T)│  ← tied
                                        └─────┬─────┘
                                        ┌─────┴─────┐
                                        │ ZCRMSNorm │
                                        └─────┬─────┘
                                     ┌────────┴────────┐
                                     │ Decoder x 8     │
                                     │┌───────────────┐│
                                     ││ ZCRMSNorm     ││
                                     ││ Masked Self   ││
                                     ││ Attn + RoPE   ││
                                     ││ Gated Residual││
                                     │├───────────────┤│
  ┌──────────────┐                   ││ ZCRMSNorm     ││
  │ Encoder x 12 │──────────────────────▶Cross Attn   ││
  │              │                   ││ Gated Residual││
  │ ┌──────────┐ │                   │└───────────────┘│
  │ │ZCRMSNorm │ │                   └────────┬────────┘
  │ │Self Attn │ │                      ┌─────┴─────┐
  │ │ GQA+RoPE │ │                      │ Embedding │  ← shared
  │ │Gated Res │ │                      └─────┬─────┘
  │ │          │ │                    ┌───────┴───────-┐
  │ │ (no FFN) │ │                    │[EOS]<tool_call>│
  │ └──────────┘ │                    │ + answer       │
  │              │                    └───────────────-┘
  └──────┬───────┘
         │
    ┌────┴──────┐
    │ Embedding │
    └────┬──────┘
         │
    ┌────┴──────┐
    │   Text    │
    │  query    │
    └───────────┘
```

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

## Citation

```
@misc{ndubuaku2026SAN,
  title={Simple Attention Networks},
  author={Henry Ndubuaku},
  year={2026},
  url={https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md}
}
```
