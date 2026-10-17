"""Numba-версии OnlineRobustSegmenter (`_robust_segment`) и морфологии маски (`close_opening`) совпадают с прежними побитно."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.transforms import square_noises as sq  # noqa: E402


def reference_process_signal(self, mask, signal):
    """Прежняя реализация `OnlineRobustSegmenter.process_signal` (чистый Python) — эталон."""
    n = len(signal)
    median_array, mad_array = [], []
    self.median_est = signal[0]
    self.mad_est = 1
    for i in range(1, n):
        if i > self.offset:
            median_array.append(self.median_est)
            mad_array.append(self.mad_est)
        sigma = self.k_mad * self.mad_est
        threshold = self.sigma_coef * sigma
        if threshold < 1e-5:
            threshold = 1e-5
        if i > self.offset:
            deviation = signal[i - self.offset] - self.median_est
            if abs(deviation) > threshold:
                mask[i - self.offset] = 1
        threshold = self.robust_sogma_coef * sigma
        if threshold < 1e-5:
            threshold = 1e-5
        x = signal[i]
        deviation = x - self.median_est
        if abs(deviation) > threshold:
            clipped_val = self.median_est + np.sign(deviation) * threshold
            self.median_est = self.alpha * clipped_val + (1 - self.alpha) * self.median_est
            input_for_mad_ema = threshold
        else:
            self.median_est = self.alpha * x + (1 - self.alpha) * self.median_est
            input_for_mad_ema = abs(deviation)
        self.mad_est = self.alpha * input_for_mad_ema + (1 - self.alpha) * self.mad_est
    median_array.append(self.median_est)
    mad_array.append(self.mad_est)
    return mask, median_array, mad_array


def signal_with_impulses(seed, n=20000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).cumsum() * 0.05 + rng.normal(0, 1, n)
    for c in rng.integers(100, n - 100, size=30):
        x[c:c + rng.integers(1, 40)] += rng.choice([-1, 1]) * rng.uniform(3, 15)
    x[5000:5200] = 0.0                                    # участок нулей: порог упирается в 1e-5
    return x


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("window,sigma_coef,robust", [(13, 2.241, 1.6098), (25, 2.0, 2.0), (2, 3.0, 1.0)])
def test_segmenter_matches_python(seed, window, sigma_coef, robust):
    x = signal_with_impulses(seed)
    mask_ref = np.zeros(len(x), dtype=bool)
    ref = sq.OnlineRobustSegmenter(window, sigma_coef, robust)
    _, med_ref, mad_ref = reference_process_signal(ref, mask_ref, x)
    mask = np.zeros(len(x), dtype=bool)
    seg = sq.OnlineRobustSegmenter(window, sigma_coef, robust)
    _, med, mad = seg.process_signal(mask, x)
    np.testing.assert_array_equal(mask, mask_ref)
    np.testing.assert_array_equal(med, np.asarray(med_ref, dtype=np.float64))
    np.testing.assert_array_equal(mad, np.asarray(mad_ref, dtype=np.float64))
    assert seg.median_est == ref.median_est and seg.mad_est == ref.mad_est
    assert mask.any() and not mask.all()


def test_ica_filtration_unchanged(monkeypatch):
    rng = np.random.default_rng(5)
    data = rng.normal(0, 1, (15, 8000)).astype(np.float32)
    data[:, 2000:2600] += np.where(np.arange(600) % 80 < 40, 6.0, -6.0)[None, :] * rng.uniform(0.5, 1.5, (15, 1))
    new = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen")
    monkeypatch.setattr(sq.OnlineRobustSegmenter, "process_signal", reference_process_signal)
    old = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen")
    np.testing.assert_array_equal(new, old)


# ------------------------------------------------------------------ морфология маски (close_opening)

def reference_close_opening(rez, close_window, open_window):
    """Прежняя реализация `close_opening` (scipy.ndimage.convolve1d) — эталон."""
    from scipy.ndimage import convolve1d
    window = np.ones(close_window, dtype=np.uint8)
    rez = convolve1d(rez.astype(np.uint8), window, axis=1, mode="nearest") >= 1
    rez = convolve1d(rez.astype(np.uint8), window, axis=1, mode="nearest") >= close_window
    window = np.ones(open_window, dtype=np.uint8)
    rez = convolve1d(rez.astype(np.uint8), window, axis=1, mode="nearest") >= open_window
    rez = convolve1d(rez.astype(np.uint8), window, axis=1, mode="nearest") >= 1
    return rez


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("close_w,open_w", [(120, 70), (13, 2), (1, 1), (2, 5), (255, 254), (300, 400)])
def test_close_opening_matches_scipy(seed, close_w, open_w):
    rng = np.random.default_rng(seed)
    n = 200 if close_w >= 255 else 5000                     # окна шире сигнала — проверка краёв
    rez = rng.random((5, n)) < rng.uniform(0.02, 0.6)
    rez[:, :3] = True                                       # единицы у краёв (mode="nearest")
    rez[1] = False
    rez[2] = True
    out = sq.close_opening(rez, close_w, open_w)
    assert out.dtype == np.bool_ and out.shape == rez.shape
    if max(close_w, open_w) <= 255:                         # шире 255 прежняя версия переполняла uint8
        np.testing.assert_array_equal(out, reference_close_opening(rez, close_w, open_w))


def test_ica_filtration_unchanged_with_scipy_morphology(monkeypatch):
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, (15, 8000)).astype(np.float32)
    data[:, 3000:3900] += np.where(np.arange(900) % 90 < 45, 5.0, -5.0)[None, :] * rng.uniform(0.5, 1.5, (15, 1))
    new = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen")
    monkeypatch.setattr(sq, "close_opening", reference_close_opening)
    monkeypatch.setattr(sq.OnlineRobustSegmenter, "process_signal", reference_process_signal)
    old = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen")
    np.testing.assert_array_equal(new, old)


# ------------------------------------------------------------------ критерий остановки FastICA

def test_ica_max_iter_as_production(monkeypatch):
    """max_iter = 100 (как в C++) передаётся в FastICA; на сходящемся сигнале результат как при 200."""
    seen = []
    orig = sq.FastICA

    class Spy(orig):
        def fit_transform(self, X, y=None):
            r = super().fit_transform(X)
            seen.append((self.max_iter, self.tol, self.n_iter_))
            return r

    monkeypatch.setattr(sq, "FastICA", Spy)
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (15, 6000)).astype(np.float32)
    data[:, 1000:1800] += np.where(np.arange(800) % 100 < 50, 5.0, -5.0)[None, :] * rng.uniform(0.5, 1.5, (15, 1))
    out100 = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen")
    out200 = sq.ICA_filtration(data, median=True, EMA=True, method="hyvarinen", max_iter=200)
    assert seen[0][:2] == (100, 0.01) and seen[1][0] == 200 and seen[0][2] < 100
    np.testing.assert_array_equal(out100, out200)
