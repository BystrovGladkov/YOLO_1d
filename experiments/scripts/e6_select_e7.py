"""E6 → E7: выбор настроек обучения режима MC по val и генерация конфигов E7.

Правило выбора (зафиксировано до просмотра итогов E6):
1. Кандидаты: e6_bs8_lr01 (эталон), e6_bs8_lr02, e6_bs4_lr01, e6_small, e6_head2 (fold 0, seed 0, без аугментаций).
2. Оценка — лучшая эпоха по val ⟨F1⟩₆ из step.csv (как выбор best.pt).
3. Побеждает максимум val ⟨F1⟩₆; кандидаты в пределах 0.1 п.п. от максимума считаются равными, среди них —
   меньший val ⟨σ⟩₆ (мс), при разнице σ < 0.3 мс — эталон (более простая конфигурация).

Конфиги E7 (TASKS.md E7):
- mc_baseline — выбранные настройки, без аугментаций (pipeline как в E2), 5 фолдов × сиды 0, 1 (пары с E2 `baseline`);
- mc_full — те же настройки + все аугментации MC (A3): вероятности E4 (`final.yaml`) в pipeline `ecg_multi`
  (A3 инверсия выключена), L1 dropout p 0.5 (ignore), L2 shuffle p 0.1, L3 electrode swap p 0.2 (R5 §3.4);
  лимит эпох — по числу шагов E4 (250 эпох × 54 шага), сиды 2, 3 (пары с E4 `final`).
    python experiments/scripts/e6_select_e7.py   → experiments/configs/mc_baseline.yaml, mc_full.yaml, e6_selection.json
"""
import csv
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "experiments" / "configs"
CANDIDATES = ["e6_bs8_lr01", "e6_bs8_lr02", "e6_bs4_lr01", "e6_small", "e6_head2"]
REFERENCE = "e6_bs8_lr01"
TRAIN_RECORDS = 144
S_STEPS_PER_EPOCH = 54  # E2/E4: ≈ 1714 отведений / batch 32


def best_val(name: str) -> dict:
    rows = list(csv.DictReader(open(REPO / "experiments" / name / "fold0_seed0" / "train1" / "step.csv")))
    sd = lambda r: float(r["sd_mean6_ms"]) if r["sd_mean6_ms"] not in ("nan", "") else math.inf  # noqa: E731
    best = max(rows, key=lambda r: (float(r["F1_mean6"]), -sd(r)))
    return {"name": name, "epochs": len(rows), "best_epoch": int(best["epoch"]),
            "val_F1_mean6": float(best["F1_mean6"]), "val_sd_mean6_ms": sd(best)}


def select(results: list) -> dict:
    top = max(r["val_F1_mean6"] for r in results)
    tied = [r for r in results if top - r["val_F1_mean6"] < 0.001]
    lowest = min(tied, key=lambda r: r["val_sd_mean6_ms"])
    ref = next((r for r in tied if r["name"] == REFERENCE), None)
    if ref is not None and ref["val_sd_mean6_ms"] - lowest["val_sd_mean6_ms"] < 0.3:
        return ref
    return lowest


def main():
    results = [best_val(n) for n in CANDIDATES]
    chosen = select(results)
    training = yaml.safe_load((CONFIGS / f"{chosen['name']}.yaml").read_text(encoding="utf-8"))["training"]
    steps = math.ceil(TRAIN_RECORDS / training["batch_size"])

    base = yaml.safe_load((CONFIGS / chosen["name"]).with_suffix(".yaml").read_text(encoding="utf-8"))
    mc_baseline = {**base, "name": "mc_baseline", "folds": [0, 1, 2, 3, 4], "seeds": [0, 1],
                   "description": f"E7: режим MC без аугментаций, настройки E6 ({chosen['name']}), пары с E2 baseline"}
    final = yaml.safe_load((CONFIGS / "final.yaml").read_text(encoding="utf-8"))
    aug = {**final["training"]["augmentations"], "pipeline": "ecg_multi"}
    aug["ecg"] = {**aug["ecg"], "invert_p": 0.0}
    aug["leads"] = {"dropout_p": 0.5, "dropout_labels": "ignore", "shuffle_p": 0.1, "electrode_swap_p": 0.2}
    full_training = {**training, "augmentations": aug, "num_workers": 4,
                     "epochs": round(250 * S_STEPS_PER_EPOCH / steps),
                     "early_stopping_epochs": training["early_stopping_epochs"]}
    mc_full = {**base, "name": "mc_full", "folds": [0, 1, 2, 3, 4], "seeds": [2, 3], "training": full_training,
               "description": f"E7: режим MC, все аугментации (A3), настройки E6 ({chosen['name']}), шаги как E4, пары с E4 final"}
    for cfg in (mc_baseline, mc_full):
        (CONFIGS / f"{cfg['name']}.yaml").write_text(
            f"# {cfg['description']}. Сгенерировано experiments/scripts/e6_select_e7.py.\n"
            + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = {"candidates": results, "chosen": chosen["name"], "rule": __doc__.split("Конфиги E7")[0].strip()}
    (REPO / "experiments" / "e6_selection.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")
    for r in results:
        print(f"{r['name']:14s} val F1 {100 * r['val_F1_mean6']:.2f} %  σ {r['val_sd_mean6_ms']:.1f} мс  "
              f"best {r['best_epoch']}/{r['epochs']}")
    print("выбрано:", chosen["name"], "; mc_full epochs =", full_training["epochs"])


if __name__ == "__main__":
    main()
