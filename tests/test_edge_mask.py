"""Тесты D5: маскирование краёв при загрузке (EdgeMask, Dataset `zones`) и исключение граничных точек."""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.delineation import DelineationMetrics  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402

REF = np.array([[1, 700, 750], [2, 850, 950], [0, 1100, 1150], [1, 1200, 1250], [2, 1350, 1450],
                [0, 1600, 1650], [1, 1700, 1750]], float)  # первая волна — QRS, последняя — QRS


def test_fixed_mask():
    mask = EdgeMask((6, 6), (12, 12))
    y = mask(torch.ones(1, 1000), (100, 500))
    assert not mask.random
    nz = torch.nonzero(y[0]).flatten()
    assert nz.min().item() == 94 and nz.max().item() == 512  # [100 − 6, 500 + 12]
    assert torch.equal(mask(torch.ones(1, 10), None), torch.ones(1, 10))


def test_random_mask_covers_range_and_clips():
    random.seed(0)
    mask = EdgeMask((0, 12), (0, 24))
    starts, ends = set(), set()
    for _ in range(500):
        s, e = mask.zone((100, 500), 1000)
        starts.add(100 - s)
        ends.add(e - 1 - 500)
    assert starts == set(range(13)) and ends == set(range(25))
    s, e = EdgeMask((12, 12), (24, 24)).zone((5, 995), 1000)
    assert (s, e) == (0, 1000)
    with pytest.raises(ValueError):
        EdgeMask((5, 2), (0, 1))


def make_dataset(tmp_path: Path, with_zones=True):
    (tmp_path / "signals").mkdir()
    (tmp_path / "labels").mkdir()
    torch.save(torch.ones(1, 1000), tmp_path / "signals" / "a.pt")
    (tmp_path / "labels" / "a.txt").write_text("1 0.3 0.2\n")
    with open(tmp_path / "zones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "annot_start", "annot_end"])
        w.writerow(["a", 200, 400])
    cfg = {"path": ".", "names": {0: "P", 1: "QRS", 2: "T"}, "train_data": "signals", "train_labels": "labels"}
    if with_zones:
        cfg["zones"] = "zones.csv"
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(tmp_path / "cfg.yaml")


class Identity:
    @staticmethod
    def forward(x, box, cls):
        return x, box, cls


def test_dataset_applies_mask_from_zones(tmp_path):
    cfg = make_dataset(tmp_path)
    ds = Dataset(cfg, Identity, "train", edge_mask=EdgeMask((10, 10), (20, 20)))
    x = ds[0][0]
    nz = torch.nonzero(x[0]).flatten()
    assert (nz.min().item(), nz.max().item()) == (190, 420)
    assert torch.all(Dataset(cfg, Identity, "train")[0][0] == 1)  # без edge_mask сигнал не меняется


def test_dataset_without_zones_key_ignores_mask(tmp_path):
    cfg = make_dataset(tmp_path, with_zones=False)
    ds = Dataset(cfg, Identity, "train", edge_mask=EdgeMask((10, 10), (20, 20)))
    assert ds.zones is None and torch.all(ds[0][0] == 1)


def counts(metrics, point):
    p = metrics.compute()["points"][point]
    return p["TP"], p["FP"], p["FN"]


def test_boundary_points_excluded():
    pred = REF.copy()
    pred[0, 1] += 3     # onset первого QRS: ошибка 6 мс (граничная точка)
    pred[-1, 2] += 4    # offset последнего QRS: ошибка 8 мс (граничная точка)
    m = DelineationMetrics(exclude_boundary_points=True)
    m.add(pred, REF)
    assert counts(m, "QRS_on") == (2, 0, 0)       # из 3 onset QRS граничный не учитывается
    assert counts(m, "QRS_off") == (2, 0, 0)
    assert m.compute()["points"]["QRS_on"]["sd_ms"] == 0  # ошибки граничных точек не вошли
    full = DelineationMetrics()
    full.add(pred, REF)
    assert counts(full, "QRS_on") == (3, 0, 0) and full.compute()["points"]["QRS_on"]["mean_ms"] == pytest.approx(2)
    # остальные точки первой/последней волны (offset первого, onset последнего QRS) учитываются
    assert counts(m, "P_on") == counts(full, "P_on")


def test_boundary_exclusion_ignores_nearby_unmatched_prediction_and_missed_point():
    # граничная точка пропущена (предсказание за пределами допуска) -> не FN; предсказание рядом с
    # граничной точкой, но сопоставленное не ей (дубль), -> не FP
    pred = np.vstack([REF[1:], [[1, 690, 750]], [[1, 702, 760]]])
    m = DelineationMetrics(exclude_boundary_points=True)
    m.add(pred, REF)
    tp, fp, fn = counts(m, "QRS_on")
    assert (tp, fn) == (2, 0) and fp == 0
    missing = DelineationMetrics(exclude_boundary_points=True)
    missing.add(REF[1:], REF)  # первый QRS не найден
    assert counts(missing, "QRS_on") == (2, 0, 0)


def test_settings_defaults():
    a = AugmentationSettings()
    assert (a.edge_margin_start, a.edge_margin_end, a.edge_margin_eval) == ([0, 12], [0, 24], [6, 12])
    assert TrainingSettings(names=["P"]).exclude_boundary_points is True
