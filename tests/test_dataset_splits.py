"""Тесты Dataset (списки сплитов, кэш меток, относительные пути) и разбиений LUDB (data/make_splits.py)."""
import csv
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.dataset import Dataset  # noqa: E402

SPLITS = REPO / "data" / "splits"
MANIFEST = REPO / "data" / "processed" / "ludb" / "manifest.csv"
NAMES = {0: "P", 1: "QRS", 2: "T"}


class Identity:
    @staticmethod
    def forward(x, box, cls):
        return x, box, cls


def make_samples(root: Path, ids, bad=()):
    """Папки signals/labels с примерами; для ID из `bad` разметка некорректна (центр > 1)."""
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(ids):
        torch.save(torch.full((1, 100), float(i)), root / "signals" / f"{sid}.pt")
        center = 1.5 if sid in bad else 0.5
        (root / "labels" / f"{sid}.txt").write_text(f"1 {center} 0.1\n")


def write_yaml(path: Path, config: dict) -> str:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_list_selects_samples_in_order_with_relative_paths(tmp_path):
    make_samples(tmp_path / "data", ["a", "b", "c", "d"])
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "train.txt").write_text("# комментарий\nc\na\n", encoding="utf-8")
    cfg = write_yaml(tmp_path / "lists" / "cfg.yaml", {
        "path": "..", "names": NAMES,
        "train_data": "data/signals", "train_labels": "data/labels", "train_list": "lists/train.txt"})
    ds = Dataset(cfg, Identity, "train")
    assert [Path(f).stem for f in ds.filenames] == ["c", "a"]
    assert list(ds.labels) == ds.filenames  # порядок labels = порядок filenames
    assert (tmp_path / "lists" / "train.txt.labels.cache").is_file()
    assert not (tmp_path / "data" / "labels" / "labels.cache").exists()
    x, cls, box, _ = ds[0]
    assert x[0, 0].item() == 2.0 and cls.tolist() == [[1.0]]


def test_missing_id_in_list_raises(tmp_path):
    make_samples(tmp_path, ["a"])
    (tmp_path / "l.txt").write_text("a\nzzz\n")
    cfg = write_yaml(tmp_path / "cfg.yaml", {"path": ".", "names": NAMES, "train_data": "signals",
                                             "train_labels": "labels", "train_list": "l.txt"})
    with pytest.raises(FileNotFoundError):
        Dataset(cfg, Identity, "train")


def test_separate_cache_per_split(tmp_path):
    """Один каталог разметки, разные списки: у каждого сплита свой набор и свой кэш."""
    make_samples(tmp_path, ["a", "b", "c"])
    (tmp_path / "tr.txt").write_text("a\nb\n")
    (tmp_path / "va.txt").write_text("c\n")
    cfg = write_yaml(tmp_path / "cfg.yaml", {
        "path": ".", "names": NAMES,
        "train_data": "signals", "train_labels": "labels", "train_list": "tr.txt",
        "val_data": "signals", "val_labels": "labels", "val_list": "va.txt"})
    train, val = Dataset(cfg, Identity, "train"), Dataset(cfg, Identity, "val")
    assert train.n == 2 and val.n == 1
    assert set(train.labels).isdisjoint(val.labels)


def test_backward_compatible_directory_mode_and_invalid_labels(tmp_path):
    make_samples(tmp_path, ["b", "a", "bad"], bad=("bad",))
    cfg = write_yaml(tmp_path / "cfg.yaml", {"names": NAMES, "train_data": str(tmp_path / "signals"),
                                             "train_labels": str(tmp_path / "labels")})
    ds = Dataset(cfg, Identity, "train")
    assert [Path(f).stem for f in ds.filenames] == ["a", "b"]  # отсортировано, некорректный исключён
    assert ds.n == 2 and list(ds.labels) == ds.filenames
    assert (tmp_path / "labels" / "labels.cache").is_file()
    assert Dataset(cfg, Identity, "train").n == 2  # повторно — из кэша


def test_cache_rebuilt_when_labels_change(tmp_path):
    make_samples(tmp_path, ["a"])
    cfg = write_yaml(tmp_path / "cfg.yaml", {"names": NAMES, "train_data": str(tmp_path / "signals"),
                                             "train_labels": str(tmp_path / "labels")})
    assert Dataset(cfg, Identity, "train").n == 1
    time.sleep(0.05)
    make_samples(tmp_path, ["e"])  # новые файлы -> папка разметки новее кэша
    cache = tmp_path / "labels" / "labels.cache"
    os.utime(cache, (time.time() - 10, time.time() - 10))
    assert Dataset(cfg, Identity, "train").n == 2


needs_splits = pytest.mark.skipif(not (SPLITS / "folds.csv").is_file() or not MANIFEST.is_file(),
                                  reason="нет data/splits или data/processed/ludb")


def read_lines(path):
    return [line for line in path.read_text(encoding="utf-8").split() if line]


@needs_splits
def test_ludb_splits_protocol():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable = {r["sample_id"] for r in rows if r["usable"] == "1"}
    group = {r["record"]: r["rhythm_group"] for r in rows}
    with open(SPLITS / "folds.csv", encoding="utf-8") as f:
        folds = {r["record"]: int(r["fold"]) for r in csv.DictReader(f)}
    assert set(folds) == set(group) and len(folds) == 200

    all_test = []
    for k in range(5):
        parts = {s: read_lines(SPLITS / f"fold_{k}" / f"{s}.txt") for s in ("train", "val", "test")}
        records = {s: {sid.split("_")[0] for sid in ids} for s, ids in parts.items()}
        # отведения одной записи не расходятся по сплитам, сплиты не пересекаются
        assert not (records["train"] & records["val"] or records["train"] & records["test"]
                    or records["val"] & records["test"])
        assert records["test"] == {r for r, f in folds.items() if f == k}
        # в сплитах только пригодные отведения, и все пригодные отведения фолда учтены
        ids = parts["train"] + parts["val"] + parts["test"]
        assert len(ids) == len(set(ids)) and set(ids) == usable
        # стратификация: в каждом тесте есть все три группы ритма
        assert {group[r] for r in records["test"]} == {"sinus", "af_flutter", "paced"}
        all_test += parts["test"]
    assert sorted(all_test) == sorted(usable)  # каждый пример ровно один раз в тесте


@needs_splits
def test_fold_yaml_loads_in_dataset():
    cfg = str(SPLITS / "fold_0.yaml")
    for split in ("train", "val", "test"):
        ds = Dataset(cfg, Identity, split)
        assert ds.n == len(read_lines(SPLITS / "fold_0" / f"{split}.txt"))
        assert ds.num_classes == 3
