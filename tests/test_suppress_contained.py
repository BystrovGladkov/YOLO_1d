"""Постобработка после NMS: удаление вторичных объектов, вложенных в главный (util.suppress_contained)."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import util  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402

CROSSING, AXLE, FLASH = 4, 3, 0
MAIN, SECONDARY = [CROSSING], [AXLE, CROSSING]


def det(*rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def run(d, thr=0.8):
    return util.suppress_contained(d, MAIN, SECONDARY, thr)


def test_nested_crossing_duplicate_removed():
    d = det([100, 500, 0.9, CROSSING], [150, 300, 0.6, CROSSING])        # короткий дубль внутри длинного
    torch.testing.assert_close(run(d), d[:1])


def test_more_confident_nested_crossing_survives():
    d = det([100, 500, 0.5, CROSSING], [150, 300, 0.9, CROSSING])        # вложенный, но увереннее
    out = run(d)
    assert out.shape[0] == 2                                            # длинный (0.5) не главный старше


def test_axle_counter_inside_crossing():
    d = det([0, 1000, 0.8, CROSSING], [100, 200, 0.7, AXLE], [950, 1100, 0.7, AXLE], [10, 50, 0.9, AXLE])
    out = run(d)
    # [100, 200] целиком внутри — удалён; [950, 1100] внутри на 1/3 — остаётся; [10, 50] увереннее главного
    torch.testing.assert_close(out, d[[0, 2, 3]])


def test_threshold_is_strict_and_other_classes_untouched():
    d = det([0, 100, 0.9, CROSSING], [20, 120, 0.5, AXLE], [10, 20, 0.5, FLASH])
    out = run(d)                                                        # доля AXLE ровно 0.8 — не больше
    assert out.shape[0] == 3
    assert run(d, thr=0.79).shape[0] == 2


def test_removed_object_does_not_suppress():
    # B (0.8) удаляется внутри A (0.9); C (0.7) лежит внутри B, но не внутри A — остаётся
    d = det([0, 100, 0.9, CROSSING], [10, 105, 0.8, CROSSING], [96, 104, 0.7, CROSSING])
    torch.testing.assert_close(run(d), d[[0, 2]])


def test_noop_cases():
    empty = det()
    assert run(empty).shape == (0, 4)
    d = det([0, 100, 0.9, FLASH], [10, 50, 0.5, FLASH])
    torch.testing.assert_close(run(d), d)
    torch.testing.assert_close(util.suppress_contained(d, [], SECONDARY), d)


def test_settings_default_disabled():
    s = TrainingSettings(names=["a"] * 6)
    assert s.contain_main_classes == [] and s.contain_threshold == 0.8
