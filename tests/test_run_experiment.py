"""Тесты M4: выбор лучшей эпохи в Trainer, колонки логов, сводка и реестр run_experiment.py."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
import run_experiment  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402
from yolo_1d.yolo_1d import Trainer  # noqa: E402

REF = [[0, 600, 650], [1, 700, 750], [2, 850, 950], [0, 1100, 1150], [1, 1200, 1250], [2, 1350, 1450]]


def bare_trainer(**settings) -> Trainer:
    """Trainer без модели и данных — для проверки логики выбора лучшей эпохи и колонок логов."""
    trainer = Trainer.__new__(Trainer)
    trainer._settings = TrainingSettings(names=["P", "QRS", "T"], **settings)
    return trainer


def test_score_val_loss_is_default_and_minimized():
    t = bare_trainer()
    better = t._score({"box_val": 1, "cls_val": 1, "dfl_val": 1})
    worse = t._score({"box_val": 2, "cls_val": 1, "dfl_val": 1})
    assert better > worse


def test_score_f1_mean6_with_sd_tiebreak():
    t = bare_trainer(delineation_metrics=True, best_metric="F1_mean6")
    assert t._score({"F1_mean6": 0.9, "sd_mean6_ms": 30}) > t._score({"F1_mean6": 0.8, "sd_mean6_ms": 10})
    assert t._score({"F1_mean6": 0.9, "sd_mean6_ms": 10}) > t._score({"F1_mean6": 0.9, "sd_mean6_ms": 20})
    assert t._score({"F1_mean6": 0.1, "sd_mean6_ms": float("nan")}) > (-np.inf, -np.inf)


def test_log_columns_unchanged_without_delineation():
    base = ['epoch', 'box_train', 'cls_train', 'dfl_train', 'box_val', 'cls_val', 'dfl_val',
            'Recall', 'Precision', 'mAP@50', 'mAP']
    assert bare_trainer()._log_fieldnames() == base
    ecg = bare_trainer(delineation_metrics=True)._log_fieldnames()
    assert ecg[:len(base)] == base and "F1_mean6" in ecg and "F1_QRS_off" in ecg


def write_run(run_dir: Path, preds, thresholds):
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "thresholds.json").write_text(json.dumps({k: {"conf": v, "val_F1": 1.0} for k, v in thresholds.items()}))
    (run_dir / "predictions_test.json").write_text(json.dumps(preds))


def test_summarize_pools_folds_and_applies_thresholds(tmp_path):
    exact = [[c, a, b, 0.9] for c, a, b in REF]
    noisy = exact + [[1, 1000, 1050, 0.2]]  # ниже порога QRS 0.5 -> отбрасывается
    missing = exact[:4] + exact[5:]  # пропущен QRS
    write_run(tmp_path / "fold0_seed0", [{"id": "1_ii", "pred": noisy, "ref": REF}], {"P": 0.5, "QRS": 0.5, "T": 0.5})
    write_run(tmp_path / "fold1_seed0", [{"id": "2_ii", "pred": missing, "ref": REF}], {"P": 0.5, "QRS": 0.5, "T": 0.5})
    config = {"name": "t", "folds": [0, 1], "seeds": [0]}
    groups = {"1_ii": "sinus", "2_ii": "af_flutter"}
    summary = run_experiment.summarize(config, tmp_path, groups)
    assert [r["fold"] for r in summary["runs"]] == [0, 1]
    assert summary["runs"][0]["F1_mean6"] == 1.0  # шумовая детекция ниже порога
    pooled = summary["pooled"][0]
    assert pooled["all"]["points"]["QRS_on"]["TP"] == 3 and pooled["all"]["points"]["QRS_on"]["FN"] == 1
    assert pooled["sinus"]["summary"]["F1_mean6"] == 1.0
    assert (tmp_path / "summary.md").is_file()


def test_registry_keeps_one_row_per_experiment(tmp_path):
    summary = {"pooled": {0: {"all": {"summary": {"F1_mean6": 0.9, "sd_mean6_ms": 12.0}}}},
               "folds_mean_sd": {"F1_mean6": [0.9, 0.01]}}
    env = {"date": "d", "git_hash": "abc", "git_dirty": False}
    for name in ("a", "b", "a"):
        run_experiment.append_registry({"name": name, "folds": [0], "seeds": [0]}, env, summary,
                                       Path("c.yaml"), tmp_path)
    with open(tmp_path / "results.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == ["b", "a"]


def test_config_validation(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\ndataset: d\nfolds: [0]\nseeds: [0]\ntraining: {seed: 1}\n", encoding="utf-8")
    args = type("A", (), {"folds": None, "seeds": None, "epochs": None})
    with pytest.raises(ValueError):
        run_experiment.load_config(path, args)
    example = run_experiment.load_config(REPO / "experiments" / "configs" / "ludb_baseline_A.yaml", args)
    settings = TrainingSettings(**example["training"])  # пример конфига валиден для TrainingSettings
    assert settings.best_metric == "F1_mean6" and settings.backbone_strides == "A"
