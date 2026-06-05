# NVFP4 GPT training (emulated)

Train a small (<300M) GPT language model with **NVFP4** 4-bit quantized GEMMs on
**FineWeb-Edu**, following the recipe from NVIDIA's *Pretraining Large Language
Models with NVFP4* (arXiv 2509.25149), and log the loss to Weights & Biases.

📊 **W&B report: [Muon with NVFP4](https://api.wandb.ai/links/lzchen-ut/xezttsbx)**

## What's here

| file | purpose |
|------|---------|
| `nvfp4.py` | Emulated NVFP4: E2M1 elements, 16-block FP8-E4M3 scales + FP32 per-tensor scale. Full training recipe — all 3 GEMMs in NVFP4, stochastic-rounded gradients, **2D 16×16 weight scaling**, and a **16×16 random Hadamard transform** on the Wgrad inputs. `NVFP4Linear` swaps in for `nn.Linear`. |
| `model.py` | 123.6M-param GPT, **LLaMA-style: RoPE + RMSNorm**. Token embedding + LM head stay BF16; `hp_last_blocks` keeps the last N transformer blocks in BF16 (selective high precision). |
| `muon.py` | **Muon** optimizer (Moonlight / Moonshot AI version): Newton-Schulz orthogonalized momentum for 2D hidden weights + built-in AdamW for embeddings/head/norms. |
| `download_fineweb.py` / `tokenize_fineweb.py` | Download FineWeb-Edu `sample/350BT` (~1 TB) and tokenize to 349B GPT-2 BPE tokens (memory-bounded, resumable). |
| `train.py` | Training loop (cosine LR, eval, wandb). `--precision {nvfp4,bf16}`, `--optimizer {adamw,muon}`, `--quant_backward`, `--hp_last_blocks`. Auto-detects single-file vs sharded data. |

## Alignment with arXiv 2509.25149

All four of the paper's recommended techniques are implemented:
1. **Selective high precision** — keep ~15% of layers in BF16, mostly at the end (`--hp_last_blocks 2` → last 2 of 12 blocks).
2. **Random Hadamard transform** (16×16) on the inputs of the weight-gradient GEMM.
3. **2D (16×16) scaling for weights** (same quantized weight in forward & backward), 1D (1×16) for activations and gradients.
4. **Stochastic rounding** for gradients, round-to-nearest-even for weights/activations.

Hardware note: NVFP4 GEMMs need Blackwell (sm_100+). This box is H100 (Hopper),
so GEMM operands are quantize→dequantized to exact NVFP4 numerics and the matmul
runs in BF16 — faithful to NVFP4 *numerics/loss*, not its speed. On Blackwell the
FP4 tensor cores make it faster than BF16.

## Run

```bash
source .venv/bin/activate
export WANDB_API_KEY=...
python download_fineweb.py && python tokenize_fineweb.py --workers 96

COMMON="--data_dir data_fineweb/bin --block_size 512 --batch_size 32 --max_iters 10000 \
  --warmup_iters 200 --eval_interval 250 --wandb_project nvfp4-paper-aligned"
# bf16 reference
python train.py --precision bf16  --optimizer adamw --lr 6e-4 $COMMON --wandb_run_name bf16-ref
# NVFP4 paper recipe + AdamW
python train.py --precision nvfp4 --quant_backward --hp_last_blocks 2 --optimizer adamw --lr 6e-4 $COMMON --wandb_run_name nvfp4-paper-adamw
# NVFP4 paper recipe + Muon
python train.py --precision nvfp4 --quant_backward --hp_last_blocks 2 --optimizer muon --lr 2e-3 $COMMON --wandb_run_name nvfp4-paper-muon
```

## Results (124M model, 10k iters, FineWeb-Edu)

| run | final val loss | runtime (1×H100) |
|-----|----------------|------------------|
| bf16 reference (AdamW) | 3.858 | 13.9 min |
| NVFP4 paper recipe + AdamW | 3.885 | 72.2 min |
| **NVFP4 paper recipe + Muon** | **3.788** | 76.3 min |

* NVFP4+AdamW tracks bf16 within **+0.028 nats (~0.7%)** — consistent with the paper's "<1%" FP4-vs-FP8 gap.
* NVFP4+**Muon** beats both NVFP4+AdamW and the bf16+AdamW reference: the optimizer gain outweighs the 4-bit quantization cost.
* (NVFP4 is slower here only because 4-bit is *emulated* on Hopper.)

Full write-up: [Muon with NVFP4 (W&B report)](https://api.wandb.ai/links/lzchen-ut/xezttsbx)
