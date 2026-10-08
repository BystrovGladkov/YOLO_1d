"""Картинки к анализу ошибок E4: худшие записи test, итоговая модель (experiments/final, seed 2).

Для каждой записи берётся модель того фолда, в test которого запись попала (data/splits/folds.csv),
и рисуются выбранные отведения: разметка эксперта и ответ модели (обозначения — как в plot_predictions.py).
    python experiments/scripts/plot_e4_errors.py  →  docs/results/e4/err_<запись>.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_predictions as pp  # noqa: E402
from plot_predictions import REPO  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402

OUT = REPO / "docs" / "results" / "e4"
SEED = 2
CASES = {  # запись → (отведения, подпись)
    "116": (["ii", "v2", "avl"], "полная блокада ножки: P видна в II, почти изоэлектрична в V2/aVL — эксперт размечает P во всех отведениях"),
    "90": (["ii", "v1"], "ЭКС, AV-блокада III ст.: P эксперт не размечал, модель находит предсердные волны"),
    "103": (["ii", "v1"], "типичное трепетание: F-волны перед QRS принимаются за P"),
    "45": (["ii", "v1"], "ЭКС, желудочковая экстрасистолия: ложные P"),
    "127": (["ii", "v5"], "миграция водителя ритма, ишемия: большая ошибка границ T"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(REPO / "data" / "splits" / "folds.csv", encoding="utf-8") as f:
        fold_of = {r["record"]: int(r["fold"]) for r in csv.DictReader(f)}
    with open(REPO / "data" / "processed" / "ludb" / "manifest.csv", encoding="utf-8") as f:
        zones = {r["sample_id"]: (int(r["annot_start"]), int(r["annot_end"])) for r in csv.DictReader(f)}
    models = {}
    for rec, (leads, caption) in CASES.items():
        k = fold_of[rec]
        fold_yaml = REPO / "data" / "splits" / f"fold_{k}.yaml"
        if k not in models:
            pp.FOLD_YAML = fold_yaml
            models[k] = pp.Model(REPO / "experiments" / "final" / f"fold{k}_seed{SEED}")
        ds = Dataset(str(fold_yaml), None, "test", edge_mask=EdgeMask((6, 6), (12, 12)))
        ids = [Path(f).stem for f in ds.filenames]
        leads = [f"{rec}_{l}" for l in leads if f"{rec}_{l}" in ids]
        fig, axes = plt.subplots(2 * len(leads), 1, figsize=(15, 2.0 * 2 * len(leads) + 0.8), sharex=True)
        for j, sid in enumerate(leads):
            i = ids.index(sid)
            x = ds.load_signal(i)
            ref = pp.boxes_to_ref(ds.labels[ds.filenames[i]], x.shape[-1])
            zone = zones.get(sid)
            ax = axes[2 * j]
            pp.draw_signal(ax, x[0].numpy(), f"{sid}: разметка эксперта", zone)
            pp.draw_intervals(ax, np.c_[ref, np.ones(len(ref))])
            pp.panel(axes[2 * j + 1], x[0].numpy(), f"{sid}: итоговая модель (fold {k}, seed {SEED})",
                     models[k].predict(x), ref, zone)
        axes[-1].set_xlabel("время, с")
        fig.suptitle(f"Запись {rec} (test фолда {k}): {caption}", fontsize=10)
        pp.legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = OUT / f"err_{rec}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    sys.exit(main())
