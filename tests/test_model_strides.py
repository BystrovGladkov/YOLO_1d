"""Тесты M2: настраиваемые страйды бэкбона/FPN, инициализация смещений, fuse.

Числа параметров и MACs сверяются с docs/research/R4_architecture.md §4.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.nets import nn  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402

L = 5120


def macs(model, x):
    total = [0]

    def hook(m, _, out):
        total[0] += out.numel() // out.shape[0] * m.in_channels // m.groups * m.kernel_size[0]

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total[0]


@pytest.mark.parametrize("template, expected", [("original", (32, 128, 256)), ("A", (4, 16, 32)),
                                                ("B", (8, 16, 32)), ("C", (4, 8, 16))])
def test_templates_give_expected_head_strides(template, expected):
    assert nn.head_strides(template) == expected
    model = nn.get_model("nano", 3, input_channels=1, input_length=L, strides=template)
    assert tuple(int(s) for s in model.stride) == expected
    model.train()
    maps = model(torch.zeros(2, 1, L))
    assert [m.shape for m in maps] == [(2, 2 * 16 + 3, L // s) for s in expected]


@pytest.mark.parametrize("template, params_m, macs_m", [("A", 1.12, 546), ("C", 1.07, 747)])
def test_params_and_macs_match_r4(template, params_m, macs_m):
    model = nn.get_model("nano", 3, input_channels=1, input_length=L, strides=template).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    assert params == pytest.approx(params_m, abs=0.01)
    assert macs(model, torch.zeros(1, 1, L)) / 1e6 == pytest.approx(macs_m, rel=0.01)


def test_template_a_has_same_parameters_as_original():
    """В варианте A меняются только шаги свёрток: имена и формы параметров — как в исходной сети."""
    shapes = lambda m: {k: v.shape for k, v in m.state_dict().items()}  # noqa: E731
    original = nn.get_model("nano", 3, input_channels=1, input_length=L)
    a = nn.get_model("nano", 3, input_channels=1, input_length=L, strides="A")
    assert shapes(original) == shapes(a)


def test_custom_stride_list_and_validation():
    model = nn.get_model("nano", 3, input_channels=1, input_length=L, strides=[2, 2, 1, 1, 1, 2, 2, 2])
    assert tuple(int(s) for s in model.stride) == (4, 16, 32)
    with pytest.raises(ValueError):
        nn.resolve_strides([2, 2, 2])
    with pytest.raises(ValueError):
        nn.resolve_strides("Z")
    with pytest.raises(ValueError):
        nn.get_model("nano", 3, input_channels=1, input_length=5000, strides="A")  # 5000 не делится на 32
    # уровни p1 и p2 одного разрешения (отношение 1): в FPN вместо понижения — Identity
    same = nn.get_model("nano", 3, input_channels=1, input_length=L, strides=[2, 1, 1, 1, 1, 1, 1, 2])
    assert tuple(int(s) for s in same.stride) == (2, 2, 4)
    assert isinstance(same.fpn.h3, torch.nn.Identity)


def test_bias_init_1d_and_legacy():
    model = nn.get_model("nano", 3, input_channels=1, input_length=L, strides="A", bias_init="1d",
                         expected_objects=24)
    anchors = L / 4 + L / 16 + L / 32
    for cls in model.head.cls:
        assert cls[-1].bias.data[:3].tolist() == pytest.approx([math.log(24 / 3 / anchors)] * 3, rel=1e-5)
    legacy = nn.get_model("nano", 3, input_channels=1, input_length=L, strides="A")
    assert legacy.head.cls[0][-1].bias.data[0].item() == pytest.approx(math.log(5 / 3 / (L / 4) ** 2), rel=1e-5)
    with pytest.raises(ValueError):
        nn.get_model("nano", 3, input_channels=1, input_length=L, bias_init="2d")


def test_fuse_conv_bn_keeps_outputs():
    torch.manual_seed(0)
    model = nn.get_model("nano", 3, input_channels=1, input_length=L, strides="A")
    model.train()
    with torch.no_grad():  # набрать нетривиальную статистику BatchNorm
        for _ in range(3):
            model(torch.randn(4, 1, L))
    model.eval()
    x = torch.randn(2, 1, L)
    with torch.no_grad():
        before = model(x)
        model.fuse()
        after = model(x)
    assert not any(hasattr(m, "bn") for m in model.modules() if type(m) is nn.Conv)
    assert torch.allclose(before, after, atol=1e-3, rtol=1e-3)


def test_settings_defaults_and_ecg():
    s = TrainingSettings(names=["a"])
    assert (s.backbone_strides, s.bias_init, s.expected_objects) == ("original", "legacy", 24.0)
    ecg = TrainingSettings(names=["P", "QRS", "T"], backbone_strides="A", bias_init="1d")
    assert ecg.backbone_strides == "A"
    assert TrainingSettings(names=["a"], backbone_strides=[2, 1, 1, 1, 2, 1, 2, 2]).backbone_strides[1] == 1
