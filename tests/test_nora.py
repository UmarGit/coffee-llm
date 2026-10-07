import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model

from coffee_parser.nora import (
    EPS, NORM_ATOL, ColumnNormalize, apply_parametrize_nora, check_invariants,
    column_norm_stats, delta_weight, norm_atol, normalize_columns, normalize_lora_A_,
)

HAS_FORK = "use_nora" in LoraConfig.__dataclass_fields__
needs_fork = pytest.mark.skipif(not HAS_FORK, reason="NoRA PEFT fork not installed")

R, ALPHA, DIM = 4, 4, 32


def toy(use_nora=None, r=R, alpha=ALPHA, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    m = torch.nn.Sequential()
    m.add_module("q_proj", torch.nn.Linear(DIM, DIM, bias=False).to(dtype))
    cfg = dict(r=r, lora_alpha=alpha, target_modules=["q_proj"], lora_dropout=0.0)
    if use_nora is not None:
        cfg["use_nora"] = use_nora
    return get_peft_model(m, LoraConfig(**cfg))


def test_norm_atol_widens_for_low_precision():
    assert norm_atol(torch.float32) == NORM_ATOL
    assert norm_atol(torch.bfloat16) > norm_atol(torch.float16) > norm_atol(torch.float32)
    assert norm_atol(torch.float32, torch.bfloat16) == norm_atol(torch.bfloat16)


@needs_fork
def test_bf16_base_rounds_nora_init_away_from_unit_norm():
    """The fork normalizes, then casts through the base dtype. Under bf16 that leaves
    columns unit-norm only to ~2e-3, while lora_A.dtype still reports fp32 -- so a
    dtype-blind tolerance would fail every bf16 run."""
    pm = toy("init", dtype=torch.bfloat16)
    layer = pm.base_model.model.q_proj
    assert layer.lora_A["default"].weight.dtype == torch.float32
    assert layer.get_base_layer().weight.dtype == torch.bfloat16

    s = column_norm_stats(pm)
    dev = max(abs(s["min"] - 1), abs(s["max"] - 1))
    assert dev > NORM_ATOL, "expected bf16 rounding to break the fp32 tolerance"
    assert dev < norm_atol(torch.float32, torch.bfloat16)
    check_invariants(pm, "nora_init")   # must not raise


def test_normalize_columns_gives_unit_norms():
    A = torch.randn(R, DIM) * 3
    n = normalize_columns(A).norm(dim=0)
    assert torch.allclose(n, torch.ones_like(n), atol=NORM_ATOL)


def test_epsilon_is_additive_not_clamped():
    """Matches the fork: norms land just under 1, never exactly 1."""
    A = torch.randn(R, DIM)
    n = normalize_columns(A).norm(dim=0)
    assert (n < 1.0).all()
    assert (1 - n).max() < NORM_ATOL


def test_normalize_lora_A_in_place():
    pm = toy()
    assert normalize_lora_A_(pm) == 1
    s = column_norm_stats(pm)
    assert abs(s["min"] - 1) < NORM_ATOL and abs(s["max"] - 1) < NORM_ATOL


def test_parametrize_matches_formula():
    """Independent implementation == closed form, which is the claim the paper rests on."""
    pm = toy()
    layer = pm.base_model.model.q_proj.lora_A["default"]
    A0 = layer.weight.detach().clone()
    apply_parametrize_nora(pm)
    assert torch.allclose(layer.weight, normalize_columns(A0), atol=1e-6)
    x = torch.randn(8, DIM)
    assert torch.allclose(layer(x), x @ normalize_columns(A0).T, atol=1e-5)


def test_parametrize_keeps_unit_norm_through_training():
    pm = toy()
    apply_parametrize_nora(pm)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, DIM)
    for _ in range(10):
        pm(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    layer = pm.base_model.model.q_proj.lora_A["default"]
    n = layer.weight.norm(dim=0)
    assert torch.allclose(n, torch.ones_like(n), atol=NORM_ATOL)


def test_column_normalize_module_matches_reference():
    A = torch.randn(R, DIM)
    assert torch.allclose(ColumnNormalize(EPS)(A), normalize_columns(A), atol=0)


@needs_fork
@pytest.mark.parametrize("mode,normalized", [(None, False), ("init", False), (True, True)])
def test_fork_delta_weight_matches_formula(mode, normalized):
    pm = toy(mode)
    layer = pm.base_model.model.q_proj
    A, B = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    with torch.no_grad():
        B.normal_(0, 0.5)
    got = layer.get_delta_weight("default")
    want = delta_weight(A.detach(), B.detach(), ALPHA, R, normalized)
    assert torch.allclose(got, want, atol=1e-6)
    other = delta_weight(A.detach(), B.detach(), ALPHA, R, not normalized)
    assert not torch.allclose(got, other, atol=1e-6)


@needs_fork
def test_fork_init_mode_normalizes_stored_parameter():
    s = column_norm_stats(toy("init"))
    assert abs(s["min"] - 1) < NORM_ATOL and abs(s["max"] - 1) < NORM_ATOL


@needs_fork
def test_fork_continuous_mode_leaves_parameter_unconstrained():
    """The fork normalizes inside forward(), so the stored weight is NOT unit-norm."""
    pm = toy(True)
    raw = column_norm_stats(pm)
    eff = column_norm_stats(pm, effective=True)
    assert abs(raw["max"] - 1) > NORM_ATOL
    assert abs(eff["min"] - 1) < NORM_ATOL and abs(eff["max"] - 1) < NORM_ATOL


@needs_fork
def test_fork_matches_parametrize_implementation():
    fork = toy(True)
    mine = toy(None)
    apply_parametrize_nora(mine)
    with torch.no_grad():
        for pm in (fork, mine):
            pm.base_model.model.q_proj.lora_B["default"].weight.fill_(0.1)
    x = torch.randn(8, DIM)
    fork.eval()
    mine.eval()
    with torch.no_grad():
        assert torch.allclose(fork(x), mine(x), atol=1e-5)


@needs_fork
@pytest.mark.parametrize("mode,method", [(None, "lora"), ("init", "nora_init"), (True, "nora")])
def test_check_invariants_passes_for_each_method(mode, method):
    out = check_invariants(toy(mode), method)
    assert out["lora_B_zero"]
    assert out["delta_weight_matches_formula"]


@needs_fork
@pytest.mark.parametrize("mode,method", [(None, "nora_init"), ("init", "lora"), (True, "lora")])
def test_check_invariants_rejects_mislabelled_runs(mode, method):
    with pytest.raises(AssertionError):
        check_invariants(toy(mode), method)


@needs_fork
@pytest.mark.parametrize("mode", [None, "init", True])
def test_merge_is_exact(mode):
    pm = toy(mode)
    with torch.no_grad():
        pm.base_model.model.q_proj.lora_B["default"].weight.normal_(0, 0.5)
    x = torch.randn(16, DIM)
    pm.eval()
    with torch.no_grad():
        before = pm(x)
    merged = pm.merge_and_unload()
    with torch.no_grad():
        after = merged(x)
    assert torch.allclose(before, after, atol=1e-5)
