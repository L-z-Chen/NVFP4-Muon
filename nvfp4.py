"""
Emulated NVFP4 quantization for training.

NVFP4 is NVIDIA's 4-bit floating-point format (introduced with Blackwell). A
tensor is quantized with TWO levels of scaling:

  * Elements are stored as E2M1 (1 sign / 2 exponent / 1 mantissa bit). The
    representable magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}; max = 6.
  * Every contiguous block of 16 elements (along the GEMM contraction dim)
    shares one scale stored in FP8 E4M3 (max = 448).
  * One additional FP32 per-tensor ("global") scale lets the E4M3 block scales
    use their full dynamic range.

Hardware-accelerated NVFP4 GEMMs require Blackwell (sm_100+). On Hopper (this
box is H100 / sm_90) we *emulate*: we quantize+dequantize the GEMM operands to
exact NVFP4 numerics and run the matmul in bf16/fp32. The loss curve this
produces matches what real NVFP4 hardware computes; only the speed differs.

Two recipes are provided on ``NVFP4Linear`` (selected by ``quant_backward``):
  * forward-only: only the forward GEMM operands are quantized; gradients use a
    straight-through estimator.
  * full (default): all three GEMMs of the layer are quantized to NVFP4 (the
    full NVFP4 pretraining recipe). Each GEMM is quantized along its own
    contraction axis, and the gradient tensor dY is quantized with STOCHASTIC
    rounding (weights/activations use round-to-nearest).

Reference recipe: NVIDIA "Pretraining LLMs with NVFP4" and the OCP MX spec.
"""

import torch
import torch.nn.functional as F

# E2M1 (FP4) and E4M3 (FP8) format limits.
F4_MAX = 6.0
F8_MAX = 448.0
BLOCK_SIZE = 16

# E2M1 representable positive magnitudes and the midpoints between them.
# Rounding a scaled value to the nearest of these reproduces E2M1 rounding.
_E2M1_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_E2M1_BOUNDARIES = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)


def _quantize_e2m1(x_abs: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Round non-negative values to the nearest E2M1 magnitude.

    x_abs is assumed already divided by its block scale, so values live in
    roughly [0, 6]. With stochastic=True, round up/down with probability set by
    the position between the two bracketing grid points (used for gradients).
    """
    grid = _E2M1_GRID.to(x_abs.device)
    boundaries = _E2M1_BOUNDARIES.to(x_abs.device)
    x_abs = x_abs.clamp(max=F4_MAX)

    if not stochastic:
        idx = torch.bucketize(x_abs, boundaries, right=False)
        return grid[idx]

    # Stochastic rounding: bracket x in [lo, hi] and round up with prob (x-lo)/(hi-lo).
    idx_hi = torch.bucketize(x_abs, grid, right=False).clamp(min=1, max=7)
    lo = grid[idx_hi - 1]
    hi = grid[idx_hi]
    span = (hi - lo).clamp(min=1e-12)
    frac = (x_abs - lo) / span
    rnd = torch.rand_like(x_abs)
    return torch.where(rnd < frac, hi, lo)


def quantize_along(x: torch.Tensor, dim: int, stochastic: bool = False) -> torch.Tensor:
    """NVFP4 quantize-dequantize x with the 16-element blocks laid along `dim`.

    Implemented with ``unflatten``/``flatten`` (cheap views) so quantizing along
    a non-last axis needs no full-tensor transpose+copy.
    """
    if dim < 0:
        dim += x.ndim
    L = x.shape[dim]
    assert L % BLOCK_SIZE == 0, f"contraction dim {L} must be a multiple of {BLOCK_SIZE}"

    xf = x.float()
    blocks = xf.unflatten(dim, (L // BLOCK_SIZE, BLOCK_SIZE))  # 16-elem axis at dim+1
    bdim = dim + 1

    # Per-tensor (global) FP32 scale: maps the largest block scale to E4M3 max.
    per_tensor_amax = blocks.abs().amax().clamp(min=1e-12)
    per_tensor_scale = per_tensor_amax / (F4_MAX * F8_MAX)

    # Per-block scale (FP32) -> quantize to E4M3 -> back to FP32.
    block_amax = blocks.abs().amax(dim=bdim, keepdim=True)
    block_scale = block_amax / F4_MAX
    block_scale_fp8 = (block_scale / per_tensor_scale).to(torch.float8_e4m3fn)
    effective_scale = block_scale_fp8.float() * per_tensor_scale

    safe_scale = effective_scale.clamp(min=1e-12)
    x_scaled = blocks / safe_scale
    sign = torch.sign(x_scaled)
    q_abs = _quantize_e2m1(x_scaled.abs(), stochastic=stochastic)
    deq = sign * q_abs * effective_scale  # zero-scale blocks -> 0
    return deq.flatten(dim, bdim).to(x.dtype)


def nvfp4_qdq(x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Quantize the LAST dim of x to NVFP4 and dequantize (no autograd)."""
    return quantize_along(x, dim=x.ndim - 1, stochastic=stochastic)


def quantize_2d(W: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """2D NVFP4 quantize-dequantize for a weight matrix [N, K], using one E4M3
    scale per 16x16 tile (+ FP32 per-tensor scale).

    Because each 16x16 tile shares a scale, the SAME quantized weight is valid
    whether the matmul contracts along K (forward) or along N (dgrad) — giving
    the forward/backward weight-representation consistency the NVFP4 paper gets
    from "two-dimensional (2D) scaling over 16x16 blocks for weights".
    """
    assert W.ndim == 2, "2D weight scaling expects a [N, K] matrix"
    N, K = W.shape
    assert N % BLOCK_SIZE == 0 and K % BLOCK_SIZE == 0
    Wf = W.float()
    tiles = Wf.reshape(N // BLOCK_SIZE, BLOCK_SIZE, K // BLOCK_SIZE, BLOCK_SIZE)

    per_tensor_amax = tiles.abs().amax().clamp(min=1e-12)
    per_tensor_scale = per_tensor_amax / (F4_MAX * F8_MAX)

    tile_amax = tiles.abs().amax(dim=(1, 3), keepdim=True)  # over each 16x16 tile
    tile_scale = tile_amax / F4_MAX
    tile_scale_fp8 = (tile_scale / per_tensor_scale).to(torch.float8_e4m3fn)
    effective_scale = tile_scale_fp8.float() * per_tensor_scale

    safe_scale = effective_scale.clamp(min=1e-12)
    x_scaled = tiles / safe_scale
    sign = torch.sign(x_scaled)
    q_abs = _quantize_e2m1(x_scaled.abs(), stochastic=stochastic)
    deq = sign * q_abs * effective_scale
    return deq.reshape(N, K).to(W.dtype)


def _build_rht(n: int = BLOCK_SIZE, seed: int = 0) -> torch.Tensor:
    """A fixed nxn random Hadamard transform: (normalized Sylvester Hadamard) @
    diag(random +/-1). Orthonormal, so applying it to both operands along a
    GEMM's contraction dim leaves the product unchanged while dispersing
    block-level outliers."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / (n ** 0.5)  # orthonormal
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return H @ torch.diag(s)


_RHT16_CPU = _build_rht(BLOCK_SIZE)
_rht_cache = {}


def _rht16(device):
    m = _rht_cache.get(device)
    if m is None:
        m = _RHT16_CPU.to(device=device, dtype=torch.float32)
        _rht_cache[device] = m
    return m


def apply_rht(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply the fixed 16x16 random Hadamard transform along `dim`, in blocks of
    16. Used on the inputs of the weight-gradient GEMM."""
    if dim < 0:
        dim += x.ndim
    L = x.shape[dim]
    assert L % BLOCK_SIZE == 0
    H = _rht16(x.device)
    bv = x.float().unflatten(dim, (L // BLOCK_SIZE, BLOCK_SIZE))  # 16-axis at dim+1
    bv = bv.movedim(dim + 1, -1)        # [..., 16]
    rot = bv @ H.t()                    # out_i = sum_j H[i,j] x_j
    rot = rot.movedim(-1, dim + 1)
    return rot.flatten(dim, dim + 1).to(x.dtype)


# --------------------------------------------------------------------------- #
# Forward-only recipe: straight-through estimator on the backward pass.
# --------------------------------------------------------------------------- #
class _FakeQuantNVFP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stochastic):
        return nvfp4_qdq(x, stochastic=stochastic)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize_nvfp4(x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """Differentiable NVFP4 fake-quant (STE backward)."""
    return _FakeQuantNVFP4.apply(x, stochastic)


# --------------------------------------------------------------------------- #
# Full recipe: all three GEMMs quantized to NVFP4, stochastic rounding on dY.
# --------------------------------------------------------------------------- #
class _NVFP4LinearFn(torch.autograd.Function):
    """Linear with NVFP4 forward AND backward GEMMs.

    forward:  Y  = Xq · Wqᵀ            (Xq, Wq quantized along K, round-to-nearest)
    dgrad:    dX = dYq · Wq            (dY quantized along N w/ stochastic rounding,
                                        W quantized along N, round-to-nearest)
    wgrad:    dW = dYqᵀ · Xq           (dY, X quantized along M=tokens; dY stochastic)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, use_rht):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.use_rht = use_rht
        cd = x.dtype
        xq = nvfp4_qdq(x, stochastic=False)        # activation: 1D along K
        wq = quantize_2d(weight, stochastic=False)  # weight: 2D 16x16 scaling
        b = bias.to(cd) if bias is not None else None
        return F.linear(xq.to(cd), wq.to(cd), b)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x = grad_w = grad_b = None

        # NVFP4 GEMMs accumulate in higher precision; emulate with bf16 tensor
        # cores (fp32 accumulate), matching the forward.
        md = torch.bfloat16

        # dgrad: dX = dY · W, contracting over N. The weight uses the SAME 2D
        # quantization as the forward (fwd/bwd weight consistency).
        if ctx.needs_input_grad[0]:
            dyq = nvfp4_qdq(grad_out, stochastic=True)       # dY: 1D along N (stochastic)
            wq = quantize_2d(weight, stochastic=False)        # weight: same 2D quant as fwd
            grad_x = torch.matmul(dyq.to(md), wq.to(md)).to(grad_out.dtype)

        # wgrad: dW = dYᵀ · X, contracting over M = tokens. A 16x16 random
        # Hadamard transform is applied to both inputs along M to disperse
        # block-level outliers (the transform cancels in the contraction).
        if ctx.needs_input_grad[1]:
            g2 = grad_out.reshape(-1, grad_out.shape[-1])    # [M, N]
            x2 = x.reshape(-1, x.shape[-1])                  # [M, K]
            if ctx.use_rht:
                g2 = apply_rht(g2, dim=0)                    # RHT along M
                x2 = apply_rht(x2, dim=0)                    # RHT along M (same transform)
            dyq_m = quantize_along(g2, dim=0, stochastic=True)   # dY: 1D along M (stochastic)
            xq_m = quantize_along(x2, dim=0, stochastic=False)   # X: 1D along M
            grad_w = torch.matmul(dyq_m.to(md).t(), xq_m.to(md)).to(weight.dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_b = grad_out.reshape(-1, grad_out.shape[-1]).sum(0).to(weight.dtype)

        return grad_x, grad_w, grad_b, None


class NVFP4Linear(torch.nn.Linear):
    """nn.Linear with NVFP4 quantized GEMMs.

    With ``quant_backward=True`` (default) the full NVFP4 pretraining recipe is
    used: forward + both backward GEMMs are computed in NVFP4, gradients use
    stochastic rounding. With ``quant_backward=False`` only the forward GEMM
    operands are quantized and gradients flow straight through (STE).

    Master weights stay high precision; the LM head / embeddings are not
    wrapped in this class.
    """

    quant_backward = True  # class-level switch; set before constructing the model
    use_rht = True         # apply random Hadamard transform to Wgrad inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_backward:
            return _NVFP4LinearFn.apply(x, self.weight, self.bias, self.use_rht)
        xq = quantize_nvfp4(x, stochastic=False)
        wq = quantize_nvfp4(self.weight, stochastic=False)
        return F.linear(xq, wq, self.bias)
