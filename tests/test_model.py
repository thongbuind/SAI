import pytest
import torch
from helpers import TINY_MODEL
from foundation_lab.architecture import ATTENTION_KERNELS
from foundation_lab.catalog import build_model

def test_causality(tiny_model):
    """Đổi token ở vị trí t chỉ được làm đổi logits từ vị trí t trở đi."""
    ids = torch.randint(3, 64, (2, 16))
    changed = ids.clone()
    changed[:, 10] = (changed[:, 10] + 1) % 61 + 3
    with torch.no_grad():
        a, b = tiny_model(ids), tiny_model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10], atol=0, rtol=0)
    assert not torch.allclose(a[:, 10:], b[:, 10:])

@pytest.mark.parametrize("preallocated", [True, False])
def test_kv_cache_matches_full_forward(tiny_model, preallocated):
    ids = torch.randint(3, 64, (2, 12))
    with torch.no_grad():
        full = tiny_model(ids)
        if preallocated:
            cache = tiny_model.init_cache(2, 12, ids.device)
            logits, cache = tiny_model.prefill(ids[:, :8], kv_cache=cache)
        else:
            # Đường beam search: prefill không cache rồi chép vào cache cấp sẵn.
            logits, present = tiny_model.prefill(ids[:, :8], kv_cache=None)
            cache = tiny_model.init_cache(2, 12, ids.device)
            for layer, (key, value) in enumerate(present):
                cache[layer][0][:, :, :8].copy_(key)
                cache[layer][1][:, :, :8].copy_(value)
        torch.testing.assert_close(logits, full[:, 7], atol=1e-5, rtol=0)
        for t in range(8, 12):
            logits = tiny_model.decode_step(ids[:, t], cache, cache_len=t)
            torch.testing.assert_close(logits, full[:, t], atol=1e-5, rtol=0)

def test_right_padding_does_not_change_real_tokens(tiny_model):
    """Hợp đồng padding phải: thêm PAD bên phải không đổi logits của token thật.

    Độ dài chuỗi khác nhau làm SDPA cộng dồn theo thứ tự khác, nên chỉ khớp tới sai số FP32 (~1e-6),
    không khớp từng bit như test causality (cùng độ dài).
    """
    ids = torch.randint(3, 64, (1, 10))
    padded = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a, b = tiny_model(ids), tiny_model(padded)
    torch.testing.assert_close(a, b[:, :10], atol=1e-5, rtol=1e-5)

def test_weight_tying_and_dev2_state_dict_keys(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.embed.weight
    keys = set(tiny_model.state_dict())
    expected = {"embed.weight", "lm_head.weight", "norm.weight"}
    for i in range(TINY_MODEL["num_layers"]):
        expected |= {
            f"blocks.{i}.gqa.wqkv.weight", f"blocks.{i}.gqa.wo.weight",
            f"blocks.{i}.ffn.gate_up.weight", f"blocks.{i}.ffn.down.weight",
            f"blocks.{i}.norm1.weight", f"blocks.{i}.norm2.weight",
        }
    assert keys == expected

@pytest.mark.parametrize("heads,kv_heads,t_q,t_k,causal", [
    (4, 4, 9, 9, True),     # MHA, causal
    (4, 2, 9, 9, True),     # GQA, causal
    (4, 1, 1, 9, False),    # decode: 1 query nhìn toàn bộ cache
    (4, 2, 3, 9, False),
])
def test_eager_kernel_matches_sdpa(heads, kv_heads, t_q, t_k, causal):
    torch.manual_seed(0)
    q = torch.randn(2, heads, t_q, 8, requires_grad=True)
    k = torch.randn(2, kv_heads, t_k, 8, requires_grad=True)
    v = torch.randn(2, kv_heads, t_k, 8, requires_grad=True)
    outputs, grads = [], []
    for name in ("sdpa", "eager"):
        for t in (q, k, v):
            t.grad = None
        out = ATTENTION_KERNELS.get(name)(q, k, v, is_causal=causal)
        out.sum().backward()
        outputs.append(out.detach())
        grads.append([t.grad.clone() for t in (q, k, v)])
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

MODULAR_MODEL = {**TINY_MODEL, "family": "SAI_modular"}

def test_modular_default_is_bit_identical_to_official():
    """Bản lắp ghép với lựa chọn mặc định phải giống model chính thức từng bit:
    khởi tạo, forward khi train (có dropout), gradient, prefill/decode."""
    models = []
    for spec in (TINY_MODEL, MODULAR_MODEL):
        torch.manual_seed(0)
        models.append(build_model({**spec, "dropout": 0.1}))
    official, modular = models
    for key, value in official.state_dict().items():
        assert torch.equal(value, modular.state_dict()[key]), key
    assert official.state_dict().keys() == modular.state_dict().keys()

    ids = torch.randint(3, 64, (2, 12))
    outputs = []
    for model in models:
        torch.manual_seed(1)
        model.train()
        logits = model(ids)
        logits.square().mean().backward()
        outputs.append((logits.detach(), {n: p.grad for n, p in model.named_parameters()}))
    assert torch.equal(outputs[0][0], outputs[1][0])
    for name, grad in outputs[0][1].items():
        assert torch.equal(grad, outputs[1][1][name]), name

    with torch.no_grad():
        results = []
        for model in models:
            model.eval()
            cache = model.init_cache(2, 12, ids.device)
            first, cache = model.prefill(ids[:, :8], kv_cache=cache)
            results.append((first, model.decode_step(ids[:, 8], cache, cache_len=8)))
    assert all(torch.equal(a, b) for a, b in zip(*results))

def test_swapping_kernel_keeps_weights_and_outputs(tiny_model):
    """Kernel không có tham số: dùng nguyên trọng số của model chính thức, đổi kernel cho cùng
    logits (trong sai số số học)."""
    eager_model = build_model({**MODULAR_MODEL, "attention_kernel": "eager"}).eval()
    eager_model.load_state_dict(tiny_model.state_dict())
    ids = torch.randint(3, 64, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(tiny_model(ids), eager_model(ids), atol=1e-5, rtol=1e-5)

def test_unknown_component_is_rejected():
    with pytest.raises(KeyError, match="attention_kernel 'nope'"):
        build_model({**MODULAR_MODEL, "attention_kernel": "nope"})

def test_official_model_rejects_modular_keys():
    from foundation_lab.core.config import ConfigError
    with pytest.raises(ConfigError, match="khoá không hợp lệ"):
        build_model({**TINY_MODEL, "attention_kernel": "eager"})
