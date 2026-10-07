"""NoRA helpers: normalization, invariant checks, and an independent reimplementation.

NoRA (Kang et al., 2026, "Normalized Low-Rank Adaptation", arXiv:2608.31036) constrains
each column of `lora_A.weight` (shape r x in_features) to unit L2 norm along the rank
dimension, so a column is a direction and all magnitude sits in `lora_B`.

EPS matches the official fork (github.com/Joluck/NoRA), which adds the epsilon to the
norm rather than clamping it. Column norms therefore land at 1 - O(EPS/||a||), not
exactly 1, so invariant checks use atol=1e-4.

Not imported by `coffee_parser/__init__.py`: this is the only module that needs torch.
"""
import torch
import torch.nn as nn

EPS = 1e-6
NORM_ATOL = 1e-4


def norm_atol(*dtypes):
    """Tolerance for the unit-norm check, set by the coarsest dtype involved.

    PEFT keeps lora_A in fp32 even under a bf16 base model, but the fork normalizes
    and then calls _move_adapter_to_device_of_base_layer, which round-trips the weights
    through the base dtype. Under bf16 that leaves columns unit-norm only to ~2e-3,
    stored in an fp32 tensor -- so the base layer dtype has to be considered too, or
    every bf16 NoRA-init run trips the assert.
    """
    return max([NORM_ATOL] + [4 * torch.finfo(d).eps for d in dtypes])


def normalize_columns(A, eps=EPS):
    return A / (A.norm(dim=0, keepdim=True) + eps)


def delta_weight(A, B, lora_alpha, r, normalized):
    """Reference formula: alpha/r * B @ (Norm(A) if normalized else A)."""
    return (lora_alpha / r) * (B @ (normalize_columns(A) if normalized else A))


def iter_lora_layers(model, adapter="default"):
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and adapter in getattr(mod, "lora_A", {}):
            yield name, mod


def normalize_lora_A_(model, adapter="default", eps=EPS):
    """NoRA-init: normalize every lora_A once, in place. Returns the layer count."""
    n = 0
    with torch.no_grad():
        for _, mod in iter_lora_layers(model, adapter):
            W = mod.lora_A[adapter].weight
            W.div_(W.norm(dim=0, keepdim=True) + eps)
            n += 1
    return n


def column_norm_stats(model, adapter="default", effective=False):
    lo, hi = float("inf"), 0.0
    for _, mod in iter_lora_layers(model, adapter):
        A = mod.lora_A[adapter].weight.detach()
        if effective:
            A = normalize_columns(A)
        n = A.norm(dim=0)
        lo, hi = min(lo, n.min().item()), max(hi, n.max().item())
    return {"min": lo, "max": hi}


def lora_B_all_zero(model, adapter="default"):
    return all((m.lora_B[adapter].weight == 0).all().item() for _, m in iter_lora_layers(model, adapter))


def check_invariants(model, method, adapter="default"):
    """Sanity asserts logged with every run. `method` is lora | nora_init | nora."""
    normalized = method == "nora"
    raw = column_norm_stats(model, adapter)
    eff = column_norm_stats(model, adapter, effective=True)
    _, first = next(iter_lora_layers(model, adapter))
    dtype = first.lora_A[adapter].weight.dtype
    base_dtype = first.get_base_layer().weight.dtype
    atol = norm_atol(dtype, base_dtype)
    unit = lambda s: abs(s["min"] - 1) < atol and abs(s["max"] - 1) < atol

    out = {"raw_colnorm": raw, "effective_colnorm": eff, "adapter_dtype": str(dtype),
           "base_dtype": str(base_dtype), "norm_atol": atol,
           "lora_B_zero": lora_B_all_zero(model, adapter)}
    if method == "nora_init":
        assert unit(raw), f"nora_init: lora_A columns not unit-norm: {raw}"
    elif method == "nora":
        # the stored parameter is unconstrained; the fork normalizes inside forward()
        assert unit(eff), f"nora: effective lora_A columns not unit-norm: {eff}"
    else:
        assert not unit(raw), f"lora: lora_A columns unexpectedly unit-norm: {raw}"

    # lora_B is zero at init, which makes both candidate formulas trivially equal,
    # so probe with a non-zero B and restore it afterwards.
    name, mod = next(iter_lora_layers(model, adapter))
    A, B = mod.lora_A[adapter].weight, mod.lora_B[adapter].weight
    out["delta_weight_checked_layer"] = name
    with torch.no_grad():
        saved = B.clone()
        B.normal_(0, 0.5)
        got = mod.get_delta_weight(adapter)
        want = delta_weight(A.detach(), B.detach(), mod.lora_alpha[adapter], mod.r[adapter], normalized)
        tol = max(1e-5, 20 * max(torch.finfo(dtype).eps, torch.finfo(base_dtype).eps))
        out["delta_weight_max_abs_diff"] = (got.float() - want.float()).abs().max().item()
        out["delta_weight_matches_formula"] = bool(torch.allclose(got.float(), want.float(), atol=tol, rtol=tol))
        B.copy_(saved)
    assert out["delta_weight_matches_formula"], f"{method}: get_delta_weight != reference formula at {name}"
    return out


class ColumnNormalize(nn.Module):
    """Independent NoRA implementation via torch parametrization, for cross-checking."""

    def __init__(self, eps=EPS):
        super().__init__()
        self.eps = eps

    def forward(self, A):
        return A / (A.norm(dim=0, keepdim=True) + self.eps)


def apply_parametrize_nora(model, adapter="default", eps=EPS):
    from torch.nn.utils import parametrize
    n = 0
    for _, mod in iter_lora_layers(model, adapter):
        parametrize.register_parametrization(mod.lora_A[adapter], "weight", ColumnNormalize(eps))
        n += 1
    return n
