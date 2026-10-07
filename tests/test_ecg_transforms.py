"""Тесты ЭКГ-аугментаций (A1): сохранность разметки, работа только в активной части, длина входа."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import build_ecg_training_transforms, build_training_transforms  # noqa: E402
from yolo_1d.utils.transforms import ecg_transforms as T  # noqa: E402

L = 5000
ACTIVE = (500, 4500)


def step_sample(beats=6, start=600, period=600):
    """Сигнал-«ступенька»: 1.0 внутри волн, 0.05 вне волн в активной части, 0 вне её (маска краёв).
    Волны каждого цикла: P (60), QRS (50), T (100) с паузами PR 40, ST 60."""
    x = torch.zeros(1, L)
    x[0, ACTIVE[0]:ACTIVE[1]] = 0.05
    boxes, cls = [], []
    for k in range(beats):
        t = start + k * period
        for c, (a, w) in enumerate([(t, 60), (t + 100, 50), (t + 210, 100)]):
            x[0, a:a + w] = 1.0
            boxes.append(((a + a + w) / 2 / L, w / L))
            cls.append([c])
    return x, torch.tensor(boxes, dtype=torch.float32), torch.tensor(cls, dtype=torch.float32)


def assert_boxes_on_steps(x, boxes, min_inside=0.9):
    """Каждый бокс лежит на «единицах»: среднее внутри ≥ min_inside, сразу за краями — фон."""
    n = x.shape[-1]
    for c, w in boxes.tolist():
        a, b = int(round((c - w / 2) * n)), int(round((c + w / 2) * n))
        inside = x[0, a + 1:b - 1].abs()
        assert inside.mean() >= min_inside, (a, b, float(inside.mean()))
        for edge in (a - 3, b + 2):
            if 0 <= edge < n:
                assert abs(x[0, edge]) < 0.5, (a, b, edge, float(x[0, edge]))


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("op", [T.TimeScale(1.0), T.HRVWarp(1.0), T.WindowWarp(1.0)])
def test_time_warps_keep_labels_on_waves(op):
    for _ in range(10):
        x, b, c = step_sample()
        y, b2, c2 = op(x, b, c)
        assert len(b2) == len(b) and c2.shape == c.shape
        assert_boxes_on_steps(y, b2)


def test_hrv_warp_changes_gaps_more_than_waves():
    x, b, c = step_sample()
    y, b2, c2 = T.HRVWarp(1.0, tp=(1.4, 1.5), segment=(1.0, 1.0), wave=(1.0, 1.0))(x, b, c)
    widths_before = (b[:, 1] * L).numpy()
    widths_after = (b2[:, 1] * y.shape[-1]).numpy()
    assert np.allclose(widths_before, widths_after, atol=1)       # волны не меняются
    # диастола растянута: 5 пауз TP по 290 отсчётов × (k_tp − 1) ∈ [0.33, 0.575] (с разбросом ±5 %)
    assert 5 * 290 * 0.33 <= y.shape[-1] - L <= 5 * 290 * 0.575


@pytest.mark.parametrize("op", [T.BaselineWander(1.0), T.PowerlineNoise(1.0), T.EMGBursts(1.0), T.WhiteNoise(1.0),
                                T.BaselineStep(1.0), T.RandomFilter(1.0), T.AmplitudeScale(1.0), T.Invert(1.0)])
def test_label_preserving_ops_keep_masked_edges_zero(op):
    x, b, c = step_sample()
    y, b2, c2 = op(x, b, c)
    assert y.shape == x.shape and torch.equal(b2, b) and torch.equal(c2, c)
    assert torch.all(y[:, :ACTIVE[0]] == 0) and torch.all(y[:, ACTIVE[1]:] == 0)
    assert not torch.equal(y, x)


def test_white_noise_snr():
    x = torch.zeros(1, L)
    x[0, 1000:4000] = torch.randn(3000)
    y, _, _ = T.WhiteNoise(1.0, snr=(20, 20))(x, torch.zeros(0, 2), torch.zeros(0, 1))
    noise = (y - x)[0, 1000:4000]
    snr = 20 * np.log10(float(x[0, 1000:4000].std() / noise.std()))
    assert snr == pytest.approx(20, abs=0.5)


def test_wave_scale_changes_only_waves():
    x, b, c = step_sample()
    y, _, _ = T.WaveScale(1.0, sigma_pt=0.5, sigma_qrs=0.5)(x, b, c)
    background = x == 0.05
    assert torch.allclose(y[background], x[background])


def test_time_mask_drops_or_trims_boxes():
    for _ in range(20):
        x, b, c = step_sample()
        y, b2, c2 = T.TimeMask(1.0, max_fraction=0.1)(x, b, c)
        assert len(b2) == len(c2) <= len(b)
        n = y.shape[-1]
        for cc, w in b2.tolist():  # оставшиеся боксы не лежат в обнулённом отрезке
            a, e = int(round((cc - w / 2) * n)), int(round((cc + w / 2) * n))
            assert (y[0, a + 1:e - 1] != 0).float().mean() > 0.5


def test_fit_to_length_keeps_all_waves_and_pads():
    fit = T.FitToLength(5120)
    x, b, c = step_sample()
    y, b2, _ = fit(x, b, c)
    assert y.shape[-1] == 5120 and len(b2) == len(b)
    assert_boxes_on_steps(y, b2)
    long, bl, cl = T.TimeScale(1.0, scale=(1.15, 1.15))(x, b, c)   # 5750 отсчётов
    y, b3, _ = fit(long, bl, cl)
    assert y.shape[-1] == 5120 and len(b3) == len(b)                # волны помещаются — не теряются
    assert_boxes_on_steps(y, b3)


@pytest.mark.skipif(not (REPO / "data" / "raw" / "nstdb" / "em.dat").is_file(), reason="NSTDB не скачан")
def test_nstdb_noise():
    x, b, c = step_sample()
    y, b2, _ = T.NSTDBNoise(1.0, snr=(10, 10), records=("ma",))(x, b, c)
    assert torch.equal(b2, b) and torch.all(y[:, :ACTIVE[0]] == 0)
    noise = (y - x)[0, ACTIVE[0]:ACTIVE[1]]
    snr = 20 * np.log10(float(x[0, ACTIVE[0]:ACTIVE[1]].std() / noise.std()))
    assert snr == pytest.approx(10, abs=0.5)


def test_full_pipeline_and_eddy_unchanged():
    ecg = {k: 1.0 for k in AugmentationSettings().ecg.model_dump() if k.endswith("_p")
           and k not in ("nstdb_p", "mosaic_p", "splice_p")}
    settings = AugmentationSettings(regular_length=5000, input_length=5120, pipeline="ecg", ecg=ecg)
    transform = build_ecg_training_transforms(settings)
    names = [type(t).__name__ for t in transform.transforms]
    assert names[0] == "HRVWarp" and names[-1] == "FitToLength" and "TimeMask" in names
    for _ in range(10):
        x, b, c = step_sample()
        y, b2, c2 = transform(x, b, c)
        assert y.shape == (1, 5120) and len(b2) == len(c2) > 0
        assert torch.all((b2[:, 0] > 0) & (b2[:, 0] < 1) & (b2[:, 1] > 0))
    # по умолчанию — прежний (вихретоковый) пайплайн
    eddy = build_training_transforms(AugmentationSettings())
    assert type(eddy.transforms[0]).__name__ == "RandomResize"
    zero = build_ecg_training_transforms(AugmentationSettings(input_length=5120, pipeline="ecg"))
    assert [type(t).__name__ for t in zero.transforms] == ["FitToLength"]


def test_dataset_splice(tmp_path):
    from yolo_1d.utils.dataset import Dataset

    (tmp_path / "signals").mkdir()
    (tmp_path / "labels").mkdir()
    for name in ("a", "b"):
        x, b, c = step_sample()
        torch.save(x, tmp_path / "signals" / f"{name}.pt")
        lines = [f"{int(cc)} {bb[0]:.8f} {bb[1]:.8f}" for cc, bb in zip(c[:, 0].tolist(), b.tolist())]
        (tmp_path / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"path": ".", "names": {0: "P", 1: "QRS", 2: "T"},
                                   "train_data": "signals", "train_labels": "labels"}), encoding="utf-8")
    ds = Dataset(str(cfg), None, "train")
    for aligned in (True, False):
        for _ in range(10):
            data, label = ds.load_splice(0, aligned=aligned)
            assert abs(data.shape[-1] - L) <= 600
            boxes = torch.as_tensor(label[:, 1:3])
            assert len(boxes) >= 6
            if aligned:  # разрез в паузах после T: все волны целые
                assert_boxes_on_steps(data, boxes)
