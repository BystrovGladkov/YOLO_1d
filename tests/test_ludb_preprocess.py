"""Тесты предобработки LUDB (yolo_1d.utils.ludb_preprocess) и совместимости с Dataset.

Тест совместимости с `Dataset` использует результат `data/prepare_ludb.py` и пропускается,
если данные не подготовлены.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.ludb import Wave  # noqa: E402
from yolo_1d.utils.ludb_preprocess import (annotated_zone, bandpass, labels_to_intervals, mask_outside,  # noqa: E402
                                           prepare_lead, unusable_reason, waves_to_labels, zscore)

PROCESSED = REPO / "data" / "processed" / "ludb"
WAVES = [Wave("P", 100, 120, 150), Wave("QRS", 200, 220, 250), Wave("T", 300, 340, 400)]


def test_zscore_per_lead():
    x = np.stack([np.linspace(0, 10, 1000), 3 + 0.5 * np.sin(np.arange(1000))]).astype(np.float32)
    z = zscore(x)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(z.std(axis=1), 1, atol=1e-4)
    flat = zscore(np.full((1, 100), 5.0, dtype=np.float32))
    assert np.all(np.isfinite(flat)) and np.allclose(flat, 0)


def test_zone_and_mask():
    zone = annotated_zone(WAVES, length=500, margin=10)
    assert zone == (90, 411)
    assert annotated_zone([Wave("QRS", 3, 5, 8)], length=10, margin=10) == (0, 10)
    x = np.ones((1, 500), dtype=np.float32)
    m = mask_outside(x, zone)
    assert m[0, :90].sum() == 0 and m[0, 411:].sum() == 0 and np.all(m[0, 90:411] == 1)


def test_mask_median_fill():
    x = np.arange(10, dtype=np.float32)[None]
    m = mask_outside(x, (2, 7), fill="median")
    assert np.all(m[0, :2] == 4) and np.all(m[0, 7:] == 4) and np.all(m[0, 2:7] == x[0, 2:7])
    with pytest.raises(ValueError):
        mask_outside(x, (2, 7), fill="mean")


def test_labels_roundtrip():
    labels = waves_to_labels(WAVES, length=5000)
    assert labels.shape == (3, 3)
    assert labels[:, 0].tolist() == [0, 1, 2]
    assert labels[0, 1] == pytest.approx(125 / 5000) and labels[0, 2] == pytest.approx(50 / 5000)
    intervals = labels_to_intervals(labels, 5000)
    assert np.allclose(intervals, [[0, 100, 150], [1, 200, 250], [2, 300, 400]])
    assert waves_to_labels([], 5000).shape == (0, 3)


def test_prepare_lead_zscore_without_mask():
    """D5: сигнал сохраняется без маски (маска — при загрузке), рядом — границы разметки."""
    rng = np.random.default_rng(0)
    signal = rng.normal(2.0, 3.0, 5000)
    sample = prepare_lead(signal, WAVES)
    assert sample.signal.shape == (1, 5000) and sample.signal.dtype == np.float32
    assert sample.annotated == (100, 400)
    expected = (signal - signal.mean()) / signal.std()  # Z-score по всему отведению
    assert np.allclose(sample.signal[0], expected, atol=1e-4)
    with pytest.raises(ValueError):
        prepare_lead(signal, [])


def test_bandpass_removes_baseline():
    t = np.arange(5000) / 500
    x = (np.sin(2 * np.pi * 10 * t) + 5 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)[None]
    y = bandpass(x, 0.5, 40)[0, 500:-500]  # без краевых переходных процессов фильтра
    clean = np.sin(2 * np.pi * 10 * t)[500:-500]
    # дрейф 0.05 Гц амплитудой 5 подавлен (остаток < 5 % его амплитуды), 10 Гц сохранён без сдвига фазы
    assert np.abs(y - clean).max() < 0.25


def test_unusable_reason():
    assert unusable_reason([], (100, 400)) is None
    assert unusable_reason([50, 450], (100, 400)) is None   # дефекты только на обнуляемых краях
    assert "1 волн" in unusable_reason([50, 200], (100, 400))


@pytest.mark.skipif(not (PROCESSED / "manifest.csv").is_file(), reason="нет data/processed/ludb")
def test_processed_files_load_in_dataset(tmp_path):
    """Подготовленные файлы читаются штатным Dataset и проходят его проверки разметки."""
    from yolo_1d.utils.dataset import Dataset

    ids = ["1_ii", "35_v1", "116_v1"]
    for sub in ("signals", "labels"):
        (tmp_path / sub).mkdir()
    for sid in ids:
        shutil.copy(PROCESSED / "signals" / f"{sid}.pt", tmp_path / "signals")
        shutil.copy(PROCESSED / "labels" / f"{sid}.txt", tmp_path / "labels")
    config = {"names": {0: "P", 1: "QRS", 2: "T"},
              "train_data": str(tmp_path / "signals"), "train_labels": str(tmp_path / "labels")}
    (tmp_path / "ludb.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    identity = lambda x, box, cls: (x, box, cls)  # noqa: E731
    ds = Dataset(str(tmp_path / "ludb.yaml"), transform=type("T", (), {"forward": staticmethod(identity)}),
                 type="train")
    assert len(ds) == 3 and len(ds.labels) == 3  # ни один файл не отброшен проверками Dataset
    x, cls, box, _ = ds[ds.filenames.index(str(tmp_path / "signals" / "1_ii.pt"))]
    assert x.shape == (1, 5000) and x.dtype == torch.float32
    assert box.shape[0] == cls.shape[0] > 0
    assert torch.all((box[:, 0] > 0) & (box[:, 0] < 1) & (box[:, 1] > 0))
