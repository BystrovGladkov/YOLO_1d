"""Конфиги E6 (смоук и настройка обучения режима MC, docs/results/E6_mc_smoke.md).

База — experiments/configs/baseline.yaml (E2, режим S, без аугментаций); меняются только параметры режима MC
и обучения. Эпохи пересчитаны так, чтобы число шагов оптимизатора совпадало с E2: в S 150 эпох × 54 шага
(≈ 1714 отведений / batch 32) ≈ 8100 шагов; в MC эпоха — 144 записи, batch 8 → 18 шагов → 450 эпох;
early stopping 40 эпох S ≈ 2160 шагов → 120 эпох MC; warmup 3 эпохи S ≈ 160 шагов → 9 эпох MC.
Также пишет отладочный сплит data/splits_debug/overfit8_mc.{txt,yaml} (те же 8 записей, что в E1).
    python experiments/scripts/make_e6_configs.py
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "experiments" / "configs"


def mc_config(name, description, dataset="data/splits/fold_{fold}_mc.yaml", **training):
    base = yaml.safe_load((CONFIGS / "baseline.yaml").read_text(encoding="utf-8"))
    base.update(name=name, description=description, dataset=dataset, folds=[0], seeds=[0])
    base["training"].update(input_channels=12, n_leads=12, batch_size=8, epochs=450, early_stopping_epochs=120,
                            warmup_epochs=9, num_workers=0)
    base["training"].update(training)
    path = CONFIGS / f"{name}.yaml"
    path.write_text(f"# E6 (режим MC, без аугментаций): {description}\n"
                    + yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def overfit_split():
    s = yaml.safe_load((REPO / "data" / "splits_debug" / "overfit8.yaml").read_text(encoding="utf-8"))
    records = []
    for line in (REPO / "data" / "splits_debug" / "overfit8.txt").read_text(encoding="utf-8").split():
        rec = line.rsplit("_", 1)[0]
        if rec not in records:
            records.append(rec)
    (REPO / "data" / "splits_debug" / "overfit8_mc.txt").write_text("\n".join(records) + "\n", encoding="utf-8")
    config = {"path": "..", "names": s["names"], "mode": "multi_lead", "zones": "processed/ludb_mc/leads.csv"}
    for split in ("train", "val", "test"):
        config.update({f"{split}_data": "processed/ludb_mc/signals", f"{split}_labels": "processed/ludb_mc/labels",
                       f"{split}_list": "splits_debug/overfit8_mc.txt"})
    (REPO / "data" / "splits_debug" / "overfit8_mc.yaml").write_text(
        f"# E6: тест на переобучение режима MC — записи {', '.join(records)} (как overfit8 E1), train = val = test.\n"
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    overfit_split()
    paths = [
        mc_config("e6_overfit8", "переобучение на 8 записях, train = val = test, 900 эпох (≈ шаги E1)",
                  dataset="data/splits_debug/overfit8_mc.yaml", epochs=900, early_stopping_epochs=0, warmup_epochs=3),
        mc_config("e6_bs8_lr01", "batch 8 записей, lr 0.01 (эталон: шаги как в E2)"),
        mc_config("e6_bs8_lr02", "batch 8 записей, lr 0.02", max_lr=0.02),
        mc_config("e6_bs4_lr01", "batch 4 записи, lr 0.01 (36 шагов/эпоху)", batch_size=4, epochs=225,
                  early_stopping_epochs=60, warmup_epochs=5),
        mc_config("e6_small", "модель small, batch 8, lr 0.01", model_size="small"),
        mc_config("e6_head2", "ширина веток головы ×2, batch 8, lr 0.01", head_width_mult=2.0),
    ]
    for p in paths:
        print(p.relative_to(REPO))


if __name__ == "__main__":
    main()
