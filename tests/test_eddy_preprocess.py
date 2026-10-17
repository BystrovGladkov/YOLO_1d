"""Тесты предобработки вихретоковых дефектограмм (yolo_1d.utils.eddy, задача D1).

Тест на реальном фрагменте ECRv1 пропускается, если датасет недоступен.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.eddy import preprocess, substitute_channels  # noqa: E402
from yolo_1d.utils.eddy.kuzmin_levels import (Get_Probability_Distribution,  # noqa: E402
                                              Get_Probability_Distribution_Fast, find_closest, ClassThreshold)

ECR_TRAIN = Path(r"E:\rails\public_dataset\dataset\train\data")


def synthetic(length=6000, seed=0, dead=()):
    """Шум рельса (~N(0, 40)) + несколько «объектов» во всех каналах; каналы из `dead` — нулевые."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 40, size=(15, length))
    for c in rng.integers(500, length - 500, size=4):
        x[:, c - 60:c + 60] += rng.normal(0, 400, size=(15, 1)) * np.hanning(120)
    x = np.round(x).astype(np.int16)
    for d in dead:
        x[d] = 0
    return x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_distribution_matches_original(seed):
    rng = np.random.default_rng(seed)
    ec = rng.normal(0, 300, size=(5000, 3))          # дробные, отрицательные, выходящие за n
    ec[::7, 1] = 0.0
    ec[::11, 2] = -0.7                                # int(-0.7) == 0
    n = 1024
    for j in range(ec.shape[1]):
        a1, m1 = np.zeros(n + 1), np.zeros(n + 1)
        a2, m2 = np.zeros(n + 1), np.zeros(n + 1)
        r1 = Get_Probability_Distribution(ec, a1, m1, n, j)
        r2 = Get_Probability_Distribution_Fast(ec, a2, m2, n, j)
        assert r1 == r2
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(m1, m2)


def test_substitute_channels():
    data = np.arange(15)[:, None] * np.ones((15, 4))
    sub = list(range(15))
    sub[3], sub[7] = 4, -1
    out = substitute_channels(data, sub)
    np.testing.assert_array_equal(out[3], data[4])
    np.testing.assert_array_equal(out[7], data[7])   # заменить нечем — без изменений
    np.testing.assert_array_equal(data[3], 3)        # исходник не изменён


def test_find_closest_prefers_lower_neighbour():
    thr = [ClassThreshold(badchl=0) for _ in range(15)]
    thr[5].badchl = 1
    thr[0].badchl = 1
    sub = find_closest(thr)
    assert sub[5] == 4 and sub[0] == 1 and sub[1] == 1


def test_preprocess_shapes_and_normalisation():
    r = preprocess(synthetic())
    assert r.raw.shape == r.filtered.shape == (15, 6000)
    assert r.raw.dtype == r.filtered.dtype == np.float32
    assert np.isfinite(r.raw).all() and np.isfinite(r.filtered).all()
    assert (r.sigma > 0).all() and not r.bad_channels and not r.sigma_fallback
    t = r.to_tensor()
    assert t.shape == (30, 6000) and t.dtype == torch.float16
    assert torch.isfinite(t).all()


def test_preprocess_dead_channel_substituted_in_both_branches():
    r = preprocess(synthetic(dead=(6,)))
    assert 6 in r.bad_channels
    src = r.substitute[6]
    assert src in (5, 7)
    np.testing.assert_array_equal(r.raw[6], r.raw[src])
    np.testing.assert_array_equal(r.filtered[6], r.filtered[src])
    assert r.sigma[6] == r.sigma[src] > 0


def test_preprocess_rejects_wrong_shape():
    with pytest.raises(ValueError):
        preprocess(np.zeros((14, 100)))


@pytest.mark.skipif(not ECR_TRAIN.exists(), reason="ECRv1 недоступен")
def test_preprocess_real_fragment():
    f = sorted(ECR_TRAIN.glob("*.npy"))[0]
    raw = np.load(f)
    r = preprocess(raw)
    assert r.raw.shape == (15, raw.shape[1])
    assert torch.isfinite(r.to_tensor()).all()
    assert (r.sigma > 0).all()
