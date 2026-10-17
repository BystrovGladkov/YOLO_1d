"""Тесты предобработки как трансформов (yolo_1d.utils.transforms.eddy_preprocess, задача D4).

Тесты на реальных фрагментах ECRv1 пропускаются, если датасет недоступен.
"""
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# модуль, а не одноимённая функция, которую реэкспортирует yolo_1d.utils.eddy
pp = importlib.import_module("yolo_1d.utils.eddy.preprocess")  # noqa: E402
from yolo_1d.utils.eddy.kuzmin_levels import Get_Signal_Thresholds  # noqa: E402
from yolo_1d.utils.transforms import eddy_preprocess as ep  # noqa: E402
from yolo_1d.utils.transforms import square_noises  # noqa: E402

ECR_TRAIN = Path(r"E:\rails\public_dataset\dataset\train\data")
real = pytest.mark.skipif(not ECR_TRAIN.exists(), reason="ECRv1 недоступен")


def synthetic(length=6000, seed=0, dead=()):
    """Шум рельса (~N(0, 40)) + несколько «объектов»; каналы из `dead` — нулевые."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 40, size=(15, length))
    for c in rng.integers(500, length - 500, size=4):
        x[:, c - 60:c + 60] += rng.normal(0, 400, size=(15, 1)) * np.hanning(120)
    x = np.round(x).astype(np.int16)
    for d in dead:
        x[d] = 0
    return x


def run(pre, x, between=None):
    """Валидационный конвейер; `between(z)` — «аугментация» между шагами 2 и 7."""
    b, c = torch.zeros(0, 2), torch.zeros(0, 1)
    y, b, c = pre.reference(torch.from_numpy(x.astype(np.float32)), b, c)
    if between is not None:
        y = between(y)
    for op in pre.validation_ops()[1:]:
        y, b, c = op(y, b, c)
    return y


@pytest.fixture
def ica_calls(monkeypatch):
    """Записывает число каналов на входе каждого вызова ICA_filtration."""
    calls = []
    orig = square_noises.ICA_filtration

    def spy(data, **kw):
        calls.append(data.shape[0])
        return orig(data, **kw)

    monkeypatch.setattr(square_noises, "ICA_filtration", spy)
    return calls


# ------------------------------------------------------------------ функции шагов

def test_detect_bad_channels_on_raw():
    bad, sub = pp.detect_bad_channels(synthetic(dead=(0, 6)).astype(np.float64))
    assert np.flatnonzero(bad).tolist() == [0, 6]
    assert sub[0] == 1 and sub[6] in (5, 7) and sub[3] == 3


def test_corrected_sigma_uses_corrected_threshold1():
    x = synthetic().astype(np.float64)
    t1, t2, c1, c2 = [], [], [], []
    Get_Signal_Thresholds(x.T, t1, t2, c1, c2)
    sigma, fallback = pp.corrected_sigma(x, list(range(15)))
    np.testing.assert_array_equal(sigma, [min(abs(t.thplus), abs(t.thmins)) / 3 for t in c1])
    assert not fallback


def test_ica_good_skips_when_few_channels(ica_calls):
    x = synthetic().astype(np.float64)
    bad = np.ones(15, dtype=bool)
    bad[:4] = False
    with pytest.warns(UserWarning, match="ICA-подавление пропущено"):
        out = pp.ica_good(x, bad)
    assert ica_calls == []
    np.testing.assert_array_equal(out, x)


# ------------------------------------------------------------------ трансформы

def test_step_shapes_and_dtypes():
    pre = ep.EddyPreprocessing()
    x = torch.from_numpy(synthetic().astype(np.float32))
    b, c = torch.zeros(1, 2), torch.zeros(1, 1)
    z, b2, c2 = pre.reference(x, b, c)
    assert z.shape == (15, 6000) and z.dtype == torch.float32 and b2 is b and c2 is c
    y, _, _ = pre.ica(z, b, c)
    assert y.shape == (30, 6000) and y.dtype == torch.float64
    torch.testing.assert_close(y[:15], z.double())           # «грязная» половина — вход без изменений
    y, _, _ = pre.destandardize(y, b, c)
    y, _, _ = pre.fourier(y, b, c)
    assert y.dtype == torch.float64
    y, _, _ = pre.normalize(y, b, c)
    assert y.shape == (30, 6000) and y.dtype == torch.float32 and torch.isfinite(y).all()
    assert pre.context.ref is None                            # контекст очищен шагом 11


def test_pipeline_equals_reference_preprocess():
    x = synthetic(seed=3)
    y = run(ep.EddyPreprocessing(), x).numpy()
    r = pp.preprocess(x)
    np.testing.assert_array_equal(y[:15], r.raw)
    np.testing.assert_array_equal(y[15:], r.filtered)


def test_steps_require_reference():
    pre = ep.EddyPreprocessing()
    with pytest.raises(RuntimeError):
        pre.ica(torch.zeros(15, 100), None, None)
    with pytest.raises(ValueError):
        pre.reference(torch.zeros(30, 100), None, None)


def test_bad_channel_excluded_from_ica_and_substituted(ica_calls):
    x = synthetic(dead=(6,))
    rng = np.random.default_rng(1)

    def revive(z):                     # «аугментация» сделала битый канал ненулевым
        z = z.clone()
        z[6] = torch.from_numpy(rng.normal(0, 1, z.shape[1]).astype(np.float32))
        return z

    pre = ep.EddyPreprocessing()
    y = run(pre, x, between=revive).numpy()
    assert ica_calls == [14, 14]       # шаг 1 и шаг 7 — без канала 6
    src = pp.detect_bad_channels(x.astype(np.float64))[1][6]
    np.testing.assert_array_equal(y[6], y[src])
    np.testing.assert_array_equal(y[15 + 6], y[15 + src])


def test_cache_same_result_and_hit(tmp_path, monkeypatch):
    x = synthetic(seed=5)
    y0 = run(ep.EddyPreprocessing(), x)
    pre = ep.EddyPreprocessing(cache_dir=str(tmp_path))
    y1 = run(pre, x)
    assert len(list((tmp_path / "reference").rglob("*.npz"))) == 1
    calls = []
    monkeypatch.setattr(ep, "reference_stats", lambda raw: calls.append(1) or pp.reference_stats(raw))
    y2 = run(pre, x)
    assert calls == []                 # опорные статистики взяты из кэша
    torch.testing.assert_close(y1, y0, rtol=0, atol=0)
    torch.testing.assert_close(y2, y0, rtol=0, atol=0)


def test_cache_skips_long_mosaic(tmp_path):
    pre = ep.EddyPreprocessing(cache_dir=str(tmp_path), cache_max_length=5000)
    run(pre, synthetic(length=6000))
    assert not (tmp_path / "reference").exists() or not list((tmp_path / "reference").rglob("*.npz"))


def test_mosaic_bad_mask_computed_on_concatenation():
    """Поведение по склейке: канал, мёртвый только в одной из частей, битым не считается
    (критерий Кузьмина смотрит распределение по всему сигналу) — он попадает в ICA."""
    a, b = synthetic(seed=1), synthetic(seed=2, dead=(6,))
    assert pp.detect_bad_channels(b.astype(np.float64))[0][6]
    bad, _ = pp.detect_bad_channels(np.concatenate([a, b], axis=1).astype(np.float64))
    assert not bad[6]


# ------------------------------------------------------------------ реальные фрагменты

@real
def test_real_fragment_pipeline_equals_reference():
    x = np.load(sorted(ECR_TRAIN.glob("*.npy"))[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = run(ep.EddyPreprocessing(), x).numpy()
        r = pp.preprocess(x)
    assert np.isfinite(y).all()
    np.testing.assert_array_equal(y[:15], r.raw)
    np.testing.assert_array_equal(y[15:], r.filtered)


@real
def test_real_fragment_with_zero_channel(ica_calls):
    x = np.load(sorted(ECR_TRAIN.glob("*.npy"))[10]).copy()
    x[4] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = run(ep.EddyPreprocessing(), x).numpy()
    assert np.isfinite(y).all()
    assert ica_calls == [14, 14]
    assert np.abs(y[4]).sum() > 0 and np.abs(y[19]).sum() > 0     # канал заменён соседом


def test_pickle_keeps_shared_context(tmp_path):
    """Воркеры DataLoader (spawn на Windows) получают трансформ через pickle: общий контекст сохраняется."""
    import pickle
    pre = ep.EddyPreprocessing(cache_dir=str(tmp_path))
    ops = pickle.loads(pickle.dumps(pre.validation_ops()))
    assert all(op.context is ops[0].context for op in ops if hasattr(op, "context"))
    assert ops[0].context is not pre.context
    x = synthetic(seed=7)
    b, c = torch.zeros(0, 2), torch.zeros(0, 1)
    y = torch.from_numpy(x.astype(np.float32))
    for op in ops:
        y, b, c = op(y, b, c)
    torch.testing.assert_close(y, run(ep.EddyPreprocessing(), x), rtol=0, atol=0)
