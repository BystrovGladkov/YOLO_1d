"""Тесты A2: правило отбраковки и парные разности абляции (experiments/ablation/ablation.py)."""
import importlib.util
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("ablation", REPO / "experiments" / "ablation" / "ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)

D_F1, D_SD = 0.0037, 0.5


def test_decide_minus():
    # Full без X: F1 заметно выше -> X отбрасывается
    assert ablation.decide(+0.005, 0.0, D_F1, D_SD, "minus") == "отбросить"
    # F1 почти тот же, σ заметно лучше без X -> отбрасывается
    assert ablation.decide(-0.001, -0.6, D_F1, D_SD, "minus") == "отбросить"
    # без X F1 хуже -> X полезна
    assert ablation.decide(-0.005, 0.0, D_F1, D_SD, "minus") == "оставить (полезна)"
    # в пределах шума -> нейтральна, сохраняется
    assert ablation.decide(+0.002, +0.2, D_F1, D_SD, "minus") == "оставить (нейтральна)"


def test_decide_plus():
    assert ablation.decide(-0.005, 0.0, D_F1, D_SD, "plus") == "вредна"
    assert ablation.decide(+0.005, 0.0, D_F1, D_SD, "plus") == "полезна"
    assert ablation.decide(+0.001, -0.6, D_F1, D_SD, "plus") == "полезна"
    assert ablation.decide(0.001, 0.1, D_F1, D_SD, "plus") == "нейтральна"


def test_paired_uses_common_runs_only():
    ref = [{"fold": f, "seed": 0, "F1_mean6": 0.98, "sd_mean6_ms": 14.0} for f in range(5)]
    cand = [{"fold": f, "seed": 0, "F1_mean6": 0.98 + 0.001 * f, "sd_mean6_ms": 13.0} for f in range(3)]
    n, df1, sdf1, dsd, sdsd = ablation.paired(ref, cand)
    assert n == 3 and math.isclose(df1, 0.001) and math.isclose(dsd, -1.0) and sdsd == 0


def test_candidate_groups_and_configs(tmp_path, monkeypatch):
    assert set(ablation.GROUPS) == {"N", "T", "A", "M", "X"} and len(ablation.CANDIDATES) == 16
    monkeypatch.setattr(ablation, "CONFIGS", tmp_path)
    monkeypatch.setattr(ablation, "BASE_CONFIG", REPO / "experiments" / "configs" / "baseline.yaml")
    path = ablation.make_config("abl_test", ["N1", "T2"], "тест", [0], [0])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    aug = config["training"]["augmentations"]
    assert aug["pipeline"] == "ecg" and aug["ecg"] == {"wander_p": 0.5, "hrv_warp_p": 0.5}
    assert config["ablation"]["augmentations"] == ["N1", "T2"]
    # всё остальное — как в baseline (num_workers влияет только на скорость загрузки, E3)
    base = yaml.safe_load((REPO / "experiments" / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    skip = {"augmentations", "num_workers"}
    assert {k: v for k, v in config["training"].items() if k not in skip} == \
           {k: v for k, v in base["training"].items() if k not in skip}
