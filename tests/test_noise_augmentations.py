"""Тесты аугментаций помех: SineNoise и SquareNoises (без подавления), задача D5."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import SineNoise, SquareNoises  # noqa: E402

B, C = torch.zeros((2, 2)), torch.zeros((2, 1))


def test_sine_shape_dtype_and_additive():
    x = torch.randn(15, 20000)
    y, b, c = SineNoise()(x.clone(), B, C)
    assert y.shape == x.shape and y.dtype == x.dtype and b is B and c is C
    assert not torch.equal(y, x)
    np.random.seed(3)
    noise_only, _, _ = SineNoise()(torch.zeros(15, 20000), B, C)
    np.random.seed(3)
    y, _, _ = SineNoise()(x.clone(), B, C)
    torch.testing.assert_close(y - x, noise_only)                 # помеха аддитивна


def test_sine_frequency_within_bounds():
    aug = SineNoise(low=0.005, high=0.02, freq_jitter=0.1)
    for seed in range(20):
        np.random.seed(seed)
        _, omega, phi, omega0 = aug.sample(15, 30000)
        assert 0.005 <= omega0 <= 0.02
        assert (omega >= omega0 * 0.9 - 1e-12).all() and (omega <= omega0 * 1.1 + 1e-12).all()
        np.testing.assert_allclose(np.diff(phi), omega[1:])      # фаза непрерывна, производная = ω(t)


def test_sine_frequency_drifts():
    np.random.seed(0)
    _, omega, _, _ = SineNoise(freq_jitter=0.1).sample(15, 50000)
    assert omega.std() > 0                                       # частота меняется по сигналу


def test_sine_channel_profile():
    aug = SineNoise(amplitude=(2.0, 2.0), sigma=(0.05, 0.05))
    coords = (np.arange(15) - 7) / 15
    for seed in range(10):
        np.random.seed(seed)
        profile, _, _, _ = aug.sample(15, 1000)
        assert profile.max() <= 2.0 + 1e-12
        k = int(profile.argmax())
        # спадает от максимума к краям
        assert (np.diff(profile[:k + 1]) >= 0).all() and (np.diff(profile[k:]) <= 0).all()
        # узкий профиль: на уровне 0.5 от максимума — не больше 3 соседних каналов
        assert (profile >= 0.5 * profile.max()).sum() <= 3
    np.random.seed(1)
    profile, _, _, _ = SineNoise(amplitude=(1.0, 1.0), sigma=(0.1, 0.1)).sample(15, 10)
    m = coords[int(profile.argmax())]
    np.testing.assert_allclose(profile, np.exp(-(coords - m) ** 2 / 0.02), atol=0.25)   # гауссиан около максимума


def test_sine_measured_frequency():
    np.random.seed(5)
    aug = SineNoise(low=0.01, high=0.01, freq_jitter=0.0, amplitude=(1, 1), sigma=(10, 10))
    y, _, _ = aug(torch.zeros(15, 40000, dtype=torch.float64), B, C)
    spec = np.abs(np.fft.rfft(y[7].numpy()))
    w = 2 * np.pi * spec.argmax() / 40000
    assert abs(w - 0.01) < 2 * np.pi / 40000


def test_sine_deterministic_with_seed():
    np.random.seed(11)
    a, _, _ = SineNoise()(torch.zeros(15, 5000), B, C)
    np.random.seed(11)
    b, _, _ = SineNoise()(torch.zeros(15, 5000), B, C)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sine_bad_bounds():
    with pytest.raises(ValueError):
        SineNoise(low=0.03, high=0.002)


def test_square_noises_only_adds_noise():
    np.random.seed(0)
    x = torch.randn(15, 30000)
    y, b, c = SquareNoises()(x.clone(), B, C)
    assert y.shape == (15, 30000) and y.dtype == x.dtype and b is B and c is C
    d = y - x
    torch.testing.assert_close(d, d[:1].expand_as(d))             # одна и та же помеха во всех каналах
    assert d.abs().max() > 1                                       # помеха есть (P(нет помехи) ≈ 0)


def test_settings_defaults_noise_off():
    a = AugmentationSettings()
    assert a.square_noises == 0 and a.sine_noise == 0
    assert (a.sine_low, a.sine_high, a.sine_amplitude, a.sine_sigma, a.sine_freq_jitter) == \
        (0.002, 0.03, [0.5, 3.0], [0.05, 0.2], 0.1)
