"""Конфиги экспериментов E2 (выбор архитектуры/размера/гиперпараметров baseline без аугментаций).

Все конфиги — копия experiments/configs/ludb_baseline_A.yaml с явными переопределениями (поле
`description` перечисляет отличия). Использование:
    python experiments/scripts/make_e2_configs.py stage1|stage2 [--base-overrides key=value ...]
"""
import argparse
import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments" / "configs" / "ludb_baseline_A.yaml"
OUT = REPO / "experiments" / "configs"

# этап 1: архитектура (страйды) и размер модели; фолды 0–1, seed 0
STAGE1 = {
    "e2s1_A_nano": {"backbone_strides": "A", "model_size": "nano"},
    "e2s1_B_nano": {"backbone_strides": "B", "model_size": "nano"},
    "e2s1_C_nano": {"backbone_strides": "C", "model_size": "nano"},
    "e2s1_A_small": {"backbone_strides": "A", "model_size": "small"},
    "e2s1_A_medium": {"backbone_strides": "A", "model_size": "medium"},
}
# этап 2: lr и batch на лучшей конфигурации этапа 1 (подставляется через --base-overrides)
STAGE2 = {
    "e2s2_ref": {},
    "e2s2_small": {"model_size": "small"},
    "e2s2_lr003": {"max_lr": 0.003},
    "e2s2_lr03": {"max_lr": 0.03},
    "e2s2_bs16": {"batch_size": 16},
    "e2s2_bs64": {"batch_size": 64},
}
SCREEN = {"folds": [0, 1], "seeds": [0]}
SCREEN_TRAINING = {"epochs": 80, "early_stopping_epochs": 20}


def parse_value(text):
    return yaml.safe_load(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--base-overrides", nargs="*", default=[], help="key=value для training (этап 2)")
    args = parser.parse_args()
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    common = {k: parse_value(v) for k, v in (o.split("=", 1) for o in args.base_overrides)}
    plan = STAGE1 if args.stage == "stage1" else STAGE2
    for name, overrides in plan.items():
        config = copy.deepcopy(base)
        config.update({"name": name, **SCREEN})
        config["training"].update({**SCREEN_TRAINING, **common, **overrides})
        diff = {**SCREEN_TRAINING, **common, **overrides}
        config["description"] = (f"E2 {args.stage}: ludb_baseline_A + {diff}; фолды {SCREEN['folds']}, "
                                 f"seed {SCREEN['seeds']}, без аугментаций.")
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
