"""Тесты чтения ECRv1 в `Dataset` (format ecr_npy, задача D3).

Тест на реальной копии пропускается, если датасет недоступен.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import ecr  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402


class Identity:
    @staticmethod
    def forward(x, box, cls):
        return x, box, cls


def make_ecr(root: Path, files: dict):
    """files: {split: {name: (length, "строки разметки")}} → папки ECRv1 + манифест."""
    for split in ecr.SPLITS:
        (root / split / "data").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, (length, text) in files.get(split, {}).items():
            sig = (np.arange(15)[:, None] * 100 + np.arange(length)[None, :] % 7).astype(np.int16)
            np.save(root / split / "data" / f"{name}.npy", sig)
            (root / split / "labels" / f"{name}.txt").write_text(text)
    return ecr.write_manifest(root, root / "cfg" / "ecr.yaml")


@pytest.fixture
def manifest(tmp_path):
    return make_ecr(tmp_path, {
        "train": {"p1_left_0001": (1000, "2 100 300\n12 500 520\n"),
                  "p1_left_0002": (1000, "1 0 1000\n5 10 20\n")},
        "val": {"p2_right_0001": (400, "3 0 200\n")},
        "test": {"p3_left_0001": (1000, "4 10 50\n")},
    })


def test_labels_converted(manifest):
    ds = Dataset(str(manifest), Identity, "train")
    assert ds.format == "ecr_npy" and ds.n == 2
    lab = ds.labels[ds.filenames[0]]
    # 2 → 1, (100, 300) → центр 0.2, ширина 0.2; 12 → 5, (500, 520) → 0.51, 0.02
    np.testing.assert_allclose(lab, [[1, 0.2, 0.2], [5, 0.51, 0.02]], atol=1e-6)
    lab2 = ds.labels[ds.filenames[1]]
    np.testing.assert_allclose(lab2, [[0, 0.5, 1.0], [4, 0.015, 0.01]], atol=1e-6)


def test_normalised_by_actual_length(manifest):
    ds = Dataset(str(manifest), Identity, "val")       # сигнал 400 отсчётов
    np.testing.assert_allclose(ds.labels[ds.filenames[0]], [[2, 0.25, 0.5]], atol=1e-6)


def test_load_signal(manifest):
    ds = Dataset(str(manifest), Identity, "train")
    x = ds.load_signal(0)
    assert x.shape == (15, 1000) and x.dtype == torch.float32
    assert x[3, 0] == 300


def test_invalid_marks_dropped(tmp_path):
    m = make_ecr(tmp_path, {"train": {"a": (100, "2 10 20\n7 1 2\n3 50 40\n1 90 120\nbad line\n2 10 20\n")}})
    with pytest.warns(UserWarning, match="некорректных отметок: 4, удалено дубликатов: 1"):
        ds = Dataset(str(m), Identity, "train")
    np.testing.assert_allclose(ds.labels[ds.filenames[0]], [[1, 0.15, 0.1]], atol=1e-6)   # дубликат удалён


def test_empty_and_missing_labels(tmp_path):
    m = make_ecr(tmp_path, {"train": {"a": (100, ""), "b": (100, "1 1 2\n")}})
    (tmp_path / "train" / "labels" / "b.txt").unlink()
    ds = Dataset(str(m), Identity, "train")
    assert ds.n == 2
    assert all(v.shape == (0, 3) for v in ds.labels.values())


def test_cache_in_preprocess_cache_and_invalidation(manifest):
    cfg = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
    ds = Dataset(str(manifest), Identity, "train")
    cache = Path(cfg["preprocess_cache"]) / "train_ecr_labels.cache"
    assert cache.exists()
    assert not (Path(cfg["train_labels"]) / "labels.cache").exists()
    assert Dataset(str(manifest), Identity, "train").labels.keys() == ds.labels.keys()
    # другая карта классов → кэш пересчитан
    cfg["class_map"] = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 12: 0}
    Path(manifest).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ds2 = Dataset(str(manifest), Identity, "train")
    assert ds2.labels[ds2.filenames[0]][:, 0].tolist() == [0, 4]


def test_getitem_mosaic_and_mixup(tmp_path):
    m = make_ecr(tmp_path, {"train": {"a": (1000, "2 100 300\n"), "b": (1000, "1 0 500\n")}})
    ds = Dataset(str(m), Identity, "train", mosaic_prob=1.0)
    x, cls, box, _ = ds[0]
    assert x.shape == (15, 2000) and len(cls) == 2
    assert (box[:, 0] <= 1).all() and (box[:, 1] <= 0.5 + 1e-6).all()
    ds = Dataset(str(m), Identity, "train", mix_up_prob=1.0)
    x, cls, box, _ = ds[0]
    assert x.shape == (15, 1000) and len(cls) == 2


def test_unknown_format(tmp_path, manifest):
    cfg = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
    cfg["format"] = "nope"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        Dataset(str(bad), Identity, "train")


REAL = REPO / "data" / "ecr_v1.yaml"


@pytest.mark.skipif(not (REAL.exists() and ecr.DEFAULT_LOCAL_ROOT.exists()), reason="ECRv1 недоступен")
# README ECRv1: 8256 / 2183 / 1207 объектов; в train и test по одному точному дубликату отметки
# (manual arc: passage82_right_0001, passage61_left_0001) — они удаляются
@pytest.mark.parametrize("split,n_files,n_objects", [("train", 2099, 8255), ("val", 560, 2183), ("test", 294, 1206)])
def test_real_dataset_counts(split, n_files, n_objects, tmp_path):
    cfg = yaml.safe_load(REAL.read_text(encoding="utf-8"))
    cfg["preprocess_cache"] = str(tmp_path)          # без кэша — разметка читается заново
    manifest = tmp_path / "ecr.yaml"
    manifest.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    ds = Dataset(str(manifest), Identity, split)
    n = sum(len(v) for v in ds.labels.values())
    assert ds.n == n_files
    assert n == n_objects
    for v in ds.labels.values():
        assert ((v[:, 1] >= 0) & (v[:, 1] <= 1) & (v[:, 2] > 0) & (v[:, 2] <= 1)).all()
