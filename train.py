"""Train a small GPT on TinyShakespeare, optionally with NVFP4 quantized GEMMs,
and log the loss to Weights & Biases.

Examples:
  python train.py --precision nvfp4 --max_iters 2000
  python train.py --precision bf16  --max_iters 2000   # baseline
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from nvfp4 import NVFP4Linear
from muon import Muon


class Batcher:
    """Samples random (x, y) windows from token .bin shards.

    Auto-detects layout in data_dir:
      * single-file:  train.bin / val.bin            (e.g. TinyShakespeare)
      * sharded:      many *.bin shards; shard 0 = val, the rest = train
                      (e.g. FineWeb-Edu, one .bin per parquet file)
    Memmaps are opened lazily and cached.
    """

    def __init__(self, data_dir, block_size, seed=None):
        self.block_size = block_size
        self._cache = {}
        # Per-rank generator so each DDP rank samples different data.
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        if os.path.exists(os.path.join(data_dir, "train.bin")):
            self.train_shards = [os.path.join(data_dir, "train.bin")]
            self.val_shards = [os.path.join(data_dir, "val.bin")]
            self.mode = "single"
        else:
            shards = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
            assert len(shards) >= 2, f"need >=2 .bin shards in {data_dir}, found {len(shards)}"
            self.val_shards = shards[:1]
            self.train_shards = shards[1:]
            self.mode = "sharded"
        print(f"data: {self.mode} | {len(self.train_shards)} train shard(s), "
              f"{len(self.val_shards)} val shard(s) from {data_dir}")

    def _mm(self, path):
        m = self._cache.get(path)
        if m is None:
            m = np.memmap(path, dtype=np.uint16, mode="r")
            self._cache[path] = m
        return m

    def get_batch(self, split, batch_size, device):
        shards = self.train_shards if split == "train" else self.val_shards
        path = shards[torch.randint(len(shards), (1,), generator=self.gen).item()]
        data = self._mm(path)
        bs = self.block_size
        ix = torch.randint(len(data) - bs, (batch_size,), generator=self.gen)
        x = torch.stack([torch.from_numpy(data[i : i + bs].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + bs].astype(np.int64)) for i in ix])
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def lr_multiplier(it, total, schedule="cosine", warmup_frac=0.1, decay_frac=0.1, min_ratio=0.1):
    """LR multiplier (applied to each optimizer's base LR).

    cosine: linear warmup over warmup_frac, then cosine decay to min_ratio.
    wsd:    Warmup-Stable-Decay — linear warmup over the first warmup_frac,
            constant (=1.0) through the stable middle, then cosine decay to
            min_ratio over the last decay_frac.
    """
    warmup = max(1, int(total * warmup_frac))
    if it < warmup:
        return (it + 1) / (warmup + 1)
    if schedule == "wsd":
        decay_start = int(total * (1.0 - decay_frac))
        if it < decay_start:
            return 1.0
        prog = min(1.0, (it - decay_start) / max(1, total - decay_start))
    else:  # cosine over everything after warmup
        prog = min(1.0, (it - warmup) / max(1, total - warmup))
    return min_ratio + 0.5 * (1.0 + math.cos(math.pi * prog)) * (1.0 - min_ratio)


class OptimGroup:
    """Wraps one or more optimizers, applying a shared LR multiplier to each
    optimizer's per-group base LR. Lets a single cosine schedule drive both
    AdamW and Muon (which use very different base LRs)."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]

    def set_lr_mult(self, m):
        for o, base in zip(self.optimizers, self.base_lrs):
            for g, b in zip(o.param_groups, base):
                g["lr"] = b * m

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)


def build_optimizer(model, args, device_type):
    """AdamW by default. With --optimizer muon, use the Moonlight Muon: the 2D
    transformer-block weight matrices are optimized with Muon and everything
    else (token embedding / LM head, RMSNorm scales) with its built-in AdamW.
    A single base --lr drives both (Muon scales it by 0.2*sqrt(max(fan)))."""
    if args.optimizer == "muon":
        muon_params, adamw_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            in_blocks = name.startswith("transformer.h")
            # Muon: 2D block weights, plus the tied embedding/LM-head if requested.
            if p.ndim == 2 and (in_blocks or args.muon_embed):
                muon_params.append(p)
            else:
                adamw_params.append(p)
        opt = Muon(lr=args.lr, wd=args.weight_decay, muon_params=muon_params,
                   momentum=args.momentum, nesterov=True, ns_steps=5,
                   adamw_params=adamw_params, adamw_betas=(0.9, 0.95))
        print(f"optimizer=muon(moonlight) | muon params={len(muon_params)} "
              f"adamw params={len(adamw_params)} | base lr={args.lr} wd={args.weight_decay}")
        return OptimGroup([opt])

    # default: AdamW, weight decay on 2D params only
    decay, nodecay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.ndim >= 2 else nodecay).append(p)
    fused = device_type == "cuda"
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=fused,
    )
    print(f"optimizer=adamw | decay={len(decay)} nodecay={len(nodecay)}")
    return OptimGroup([adamw])


@torch.no_grad()
def estimate_loss(model, batcher, eval_iters, batch_size, device, ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batcher.get_batch(split, batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=["nvfp4", "bf16"], default="nvfp4")
    p.add_argument("--quant_backward", action=argparse.BooleanOptionalAction, default=True,
                   help="(nvfp4) quantize the backward GEMMs too, with stochastic-rounded "
                        "gradients (full NVFP4 recipe). --no-quant_backward = forward-only + STE.")
    p.add_argument("--hp_last_blocks", type=int, default=0,
                   help="(nvfp4) keep this many final transformer blocks in BF16 "
                        "(paper keeps ~15%% of layers, mostly at the end).")
    p.add_argument("--rht", action=argparse.BooleanOptionalAction, default=True,
                   help="(nvfp4 full) apply 16x16 random Hadamard transform to the "
                        "weight-gradient GEMM inputs. --no-rht disables it.")
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=6e-4, help="AdamW base LR (embedding/head/norms when muon)")
    p.add_argument("--muon_lr", type=float, default=0.02, help="(unused with Moonlight Muon; base --lr drives both branches)")
    p.add_argument("--momentum", type=float, default=0.95, help="Muon momentum")
    p.add_argument("--muon_embed", action=argparse.BooleanOptionalAction, default=False,
                   help="(muon) also optimize the token embedding / LM head (the tied 2D "
                        "weight) with Muon instead of AdamW.")
    p.add_argument("--min_lr_ratio", type=float, default=0.1, help="cosine decays LR to this fraction of base")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--lr_schedule", choices=["cosine", "wsd"], default="cosine",
                   help="wsd = warmup / stable / cosine-decay (warmup_frac / 1-warmup-decay / decay_frac)")
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--decay_frac", type=float, default=0.1, help="(wsd) fraction of steps in the final decay")
    p.add_argument("--tokens_per_param", type=float, default=0.0,
                   help=">0 sets total steps from a token budget = tokens_per_param * n_params")
    p.add_argument("--global_batch_size", type=int, default=0,
                   help="total sequences per optimizer step (0 = batch_size * world_size)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "data"),
                   help="dir with train.bin/val.bin (single) or many *.bin shards (sharded)")
    p.add_argument("--wandb_project", default="nvfp4-gpt")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    # ---- DDP setup (no-op when not launched via torchrun) ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank, local_rank, world_size, device, master = 0, 0, 1, args.device, True
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    torch.manual_seed(args.seed)             # identical init across ranks
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    NVFP4Linear.quant_backward = args.quant_backward
    NVFP4Linear.use_rht = args.rht

    config = GPTConfig(
        block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout,
        quant=(args.precision == "nvfp4"), hp_last_blocks=args.hp_last_blocks,
    )
    raw_model = GPT(config).to(device)
    n_params = raw_model.num_params()

    # global batch / gradient accumulation / token budget
    global_batch = args.global_batch_size or (args.batch_size * world_size)
    assert global_batch % (args.batch_size * world_size) == 0
    grad_accum = global_batch // (args.batch_size * world_size)
    tokens_per_step = global_batch * args.block_size
    max_iters = (int(round(args.tokens_per_param * n_params / tokens_per_step))
                 if args.tokens_per_param > 0 else args.max_iters)

    suffix = ("-full" if args.quant_backward else "-fwd") if args.precision == "nvfp4" else ""
    run_name = args.wandb_run_name or f"{args.precision}{suffix}-{args.optimizer}-L{args.n_layer}"
    use_wandb = (not args.no_wandb) and master
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name,
                   config={**vars(args), "world_size": world_size, "global_batch": global_batch,
                           "grad_accum": grad_accum, "max_iters": max_iters,
                           "tokens_per_step": tokens_per_step, "n_params_m": round(n_params / 1e6, 2)})

    if master:
        if config.quant:
            bw = "forward-only"
            if args.quant_backward:
                bw = "full+SR+2D" + ("+RHT" if args.rht else " (no RHT)")
            print(f"NVFP4: quantizing {config.n_layer - config.hp_last_blocks}/{config.n_layer} "
                  f"blocks ({config.hp_last_blocks} final BF16), backward={bw}")
        print(f"DDP={ddp} world_size={world_size} | per-gpu={args.batch_size} accum={grad_accum} "
              f"global_batch={global_batch} seq={args.block_size} | {tokens_per_step:,} tok/step")
        print(f"params={n_params/1e6:.1f}M | tok/param={args.tokens_per_param} -> {max_iters:,} iters "
              f"({max_iters*tokens_per_step/1e9:.2f}B tok) | sched={args.lr_schedule} compile={args.compile}")

    batcher = Batcher(args.data_dir, args.block_size, seed=args.seed + rank)

    # optimizer built from the raw model so param-name routing works
    optimizer = build_optimizer(raw_model, args, device_type)

    model = raw_model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model.train()
    t0 = time.time()
    for it in range(max_iters + 1):
        mult = lr_multiplier(it, max_iters, args.lr_schedule, args.warmup_frac, args.decay_frac, args.min_lr_ratio)
        optimizer.set_lr_mult(mult)
        lr = args.lr * mult

        if it % args.eval_interval == 0:
            if master:
                losses = estimate_loss(raw_model, batcher, args.eval_iters, args.batch_size, device, ctx)
                dt = time.time() - t0
                print(f"iter {it:6d}/{max_iters} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                      f"| lr {lr:.2e} | {dt:.0f}s")
                if use_wandb:
                    wandb.log({"iter": it, "train/loss": losses["train"],
                               "val/loss": losses["val"], "lr": lr}, step=it)
            if ddp:
                dist.barrier()

        if it == max_iters:
            break

        for micro in range(grad_accum):
            X, Y = batcher.get_batch("train", args.batch_size, device)
            if ddp:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with ctx:
                _, loss = model(X, Y)
            (loss / grad_accum).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if master and use_wandb and it % args.log_interval == 0:
            wandb.log({"iter": it, "train/loss_step": loss.item(), "lr": lr}, step=it)

    if master:
        print("done.")
    if use_wandb:
        wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
