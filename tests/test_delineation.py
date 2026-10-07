"""Тесты метрик делинеации (yolo_1d.utils.delineation), включая тест-кейсы R2 §3.2."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import (DelineationMetrics, apply_thresholds, best_class_thresholds,  # noqa: E402
                                       boxes_to_intervals, class_zones, detections_to_intervals,
                                       filter_to_zone, format_table, match_points)

# одно «отведение»: 3 кардиоцикла P-QRS-T (в отсчётах 500 Гц)
REF = np.array([[0, 600, 650], [1, 700, 750], [2, 850, 950],
                [0, 1100, 1150], [1, 1200, 1250], [2, 1350, 1450],
                [0, 1600, 1650], [1, 1700, 1750], [2, 1850, 1950]], float)


def result_of(pred, ref=REF, **kw):
    m = DelineationMetrics(**kw)
    m.add(pred, ref)
    return m.compute()


def test_perfect_prediction():
    r = result_of(REF)
    for p in r["points"].values():
        assert (p["Se"], p["PPV"], p["F1"], p["mean_ms"], p["sd_ms"]) == (1, 1, 1, 0, 0)
    assert r["summary"]["F1_mean6"] == 1 and r["summary"]["sd_mean6_ms"] == 0
    assert all(w["F1"] == 1 for w in r["waves"].values())


def test_shift_10ms_gives_mean_10_sd_0():
    pred = REF.copy()
    pred[:, 1:] += 5  # 5 отсчётов = 10 мс
    r = result_of(pred)
    for p in r["points"].values():
        assert p["F1"] == 1 and p["mean_ms"] == pytest.approx(10) and p["sd_ms"] == pytest.approx(0)


def test_shift_beyond_tolerance():
    """Сдвиг 151 мс: ни одна точка не в допуске 150 мс. Исключение краёв выключено, чтобы
    сдвинутые за зону детекции не игнорировались, а считались FP."""
    pred = REF.copy()
    pred[:, 1:] += 75.5  # 151 мс
    r = result_of(pred, exclude_edges=False)
    for p in r["points"].values():
        assert p["Se"] == 0 and p["PPV"] == 0 and p["F1"] == 0


def test_tolerance_boundary_is_inclusive():
    pairs, err = match_points([175], [100], tolerance=75)  # ровно 150 мс
    assert len(pairs) == 1 and err[0] == 75


def test_duplicate_detection_lowers_ppv_only():
    pred = np.vstack([REF, REF[1] + [0, 3, 3]])  # дубль QRS на 6 мс правее
    r = result_of(pred)
    q = r["points"]["QRS_on"]
    assert q["Se"] == 1 and q["PPV"] == pytest.approx(3 / 4) and q["FP"] == 1
    assert q["mean_ms"] == 0  # лучшая пара — точное совпадение (жадно по |ошибке|)
    assert r["points"]["P_on"]["PPV"] == 1


def test_one_to_one_greedy_matching():
    pairs, err = match_points([100, 104], [102], tolerance=75)
    assert len(pairs) == 1  # одна точка разметки подтверждает только одно предсказание
    pairs, err = match_points([100, 110], [101, 112], tolerance=75)
    assert sorted(err.tolist()) == [-2, -1]


def test_missing_wave_is_fn():
    r = result_of(np.delete(REF, 4, axis=0))  # пропущен второй QRS
    q = r["points"]["QRS_off"]
    assert q["FN"] == 1 and q["Se"] == pytest.approx(2 / 3) and q["PPV"] == 1
    assert r["waves"]["QRS"]["FN"] == 1


def test_edge_detections_are_ignored():
    """Неразмеченный кардиоцикл перед первой размеченной волной не даёт FP."""
    extra = np.array([[0, 100, 150], [1, 200, 250], [2, 350, 450]], float)
    pred = np.vstack([extra, REF])
    assert result_of(pred)["summary"]["F1_mean6"] == 1
    no_excl = result_of(pred, exclude_edges=False)
    assert no_excl["points"]["QRS_on"]["FP"] == 1


def test_class_without_reference_uses_whole_zone():
    """Нет P (ФП): ложные P внутри записи — FP, на неразмеченном краю — игнорируются."""
    ref = REF[REF[:, 0] != 0]
    zones = class_zones(ref)
    assert zones[0] == (700, 1950)
    pred = np.vstack([ref, [[0, 1000, 1050], [0, 100, 150]]])
    assert len(filter_to_zone(pred, ref)) == len(ref) + 1
    p = result_of(pred, ref)["points"]["P_on"]
    assert p["FP"] == 1 and p["TP"] == 0 and math.isnan(p["Se"]) and p["PPV"] == 0


def test_pooling_groups_and_merge():
    a, b = DelineationMetrics(), DelineationMetrics()
    a.add(REF, REF, group="sinus")
    b.add(np.delete(REF, 4, axis=0), REF, group="af_flutter")
    a.merge(b)
    assert a.groups == ["af_flutter", "sinus"]
    q = a.compute()["points"]["QRS_on"]
    assert (q["TP"], q["FN"]) == (5, 1)  # micro-пулинг по двум отведениям
    assert a.compute("sinus")["points"]["QRS_on"]["Se"] == 1


def test_coordinate_conversions():
    # разметка Dataset после паддинга 5000 -> 5120: (center, width) нормированы на 5120
    cls, boxes = np.array([1]), np.array([[(700 + 750) / 2 / 5120, 50 / 5120]])
    assert boxes_to_intervals(cls, boxes, 5120).tolist() == [[1, 700, 750]]
    # выход NMS [x1, x2, conf, cls]; сдвиг паддинга 20, обрезка по 5000:
    # вторая детекция целиком в паддинге (отбрасывается), третья частично (обрезается)
    det = np.array([[720, 770, 0.9, 1], [5030, 5100, 0.5, 2], [5000, 5040, 0.4, 0]])
    out = detections_to_intervals(det, pad_offset=20, signal_length=5000)
    assert out.tolist() == [[1, 700, 750, 0.9], [0, 4980, 5000, 0.4]]


def test_threshold_selection():
    low = np.c_[REF, np.full(len(REF), 0.9)]
    noise = np.array([[1, 1000, 1040, 0.3], [2, 1500, 1580, 0.2]])
    val = [(np.vstack([low, noise]), REF)]
    thr = best_class_thresholds(val)
    assert thr["QRS"][0] > 0.3 and thr["QRS"][1] == 1 and thr["T"][0] > 0.2
    kept = apply_thresholds(np.vstack([low, noise]), thr)
    assert len(kept) == len(REF)


def test_format_table():
    text = format_table(result_of(REF))
    assert "| QRS_on | 100.00 | 100.00 | 100.00 | 0.0 ± 0.0 |" in text and "⟨F1⟩₆ = 100.00 %" in text
