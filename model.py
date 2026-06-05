"""A small nanoGPT-style GPT with a modern (LLaMA-style) architecture:
rotary position embeddings (RoPE) and RMSNorm instead of learned absolute
position embeddings and LayerNorm. The Linear class used inside the transformer
blocks is swappable so the same model can train in plain bf16 or with NVFP4
quantized GEMMs.

Following common low-precision practice, the token embedding and the final LM
head stay in high precision (bf16); only the per-block attention and MLP
projections are quantized to NVFP4.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4 import NVFP4Linear


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304  # GPT-2 BPE (50257) padded up to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    quant: bool = False  # if True, use NVFP4Linear in the transformer blocks
    hp_last_blocks: int = 0  # keep this many final blocks in high precision (BF16)


def make_linear(quant: bool):
    return NVFP4Linear if quant else nn.Linear


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(head_dim: int, seq_len: int, theta: float):
    """Return (cos, sin) tables of shape [seq_len, head_dim] for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)            # [seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)     # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: [B, n_head, T, head_dim]; cos, sin: [T, head_dim]
    out_dtype = q.dtype
    q, k = q.float(), k.float()
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q.to(out_dtype), k.to(out_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, quant: bool):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        Linear = make_linear(quant)
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, quant: bool):
        super().__init__()
        Linear = make_linear(quant)
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, quant: bool):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, quant)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = MLP(config, quant)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                Block(config, quant=config.quant and i < config.n_layer - config.hp_last_blocks)
                for i in range(config.n_layer)
            ]),
            ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        # LM head kept in high precision; weights tied with the token embedding.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # RoPE cos/sin tables (non-persistent buffers; move with .to(device)).
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()  # tied with lm_head
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
