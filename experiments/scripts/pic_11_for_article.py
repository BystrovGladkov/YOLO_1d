"""Одно отведение: разметка эксперта, где P размечена, хотя на сигнале её почти нет, и ответ итоговой модели
(experiments/final, модель того фолда, в test которого попала запись). Обозначения — как в plot_predictions.py.
Пример: запись 116, полная блокада ножки — P видна в II, но почти изоэлектрична в V2/aVL.
    python experiments/scripts/plot_expert_p.py  →  docs/results/e4/expert_p_<отведение>.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_predictions as pp  # noqa: E402  (только Model)
from plot_predictions import REPO  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402

OUT = REPO / "docs" / "results" / "e4"
SID = "116_v2"          # отведение: "<запись>_<отведение>"
SEED = 2
T_RANGE = None          # показываемый отрезок, с, например (0.3, 9.0); None — вся запись
FS, TOL = 500, 75  # Гц, допуск сопоставления 150 мс в отсчётах
NAMES = ["P", "QRS", "T"]
COLORS = ["tab:green", "tab:red", "tab:blue"]


# ---------------------------------------------------------------- разметка и сопоставление (копия plot_predictions.py)
def boxes_to_ref(label: np.ndarray, n: int) -> np.ndarray:
    """Разметка датасета (m, 3) [класс, центр, ширина] (доли длины) → (m, 3) [класс, начало, конец] в отсчётах."""
    label = np.asarray(label, float).reshape(-1, 3)
    return np.stack([label[:, 0], (label[:, 1] - label[:, 2] / 2) * n, (label[:, 1] + label[:, 2] / 2) * n], 1)


def match(pred: np.ndarray, ref: np.ndarray):
    """Упрощённое сопоставление волн по центрам (≤ 150 мс, один-к-одному, по классам).
    Returns: множества индексов совпавших pred и ref, ошибки границ (мс) совпавших пар."""
    tp_p, tp_r, err = set(), set(), []
    for c in range(3):
        ip, ir = np.where(pred[:, 0] == c)[0], np.where(ref[:, 0] == c)[0]
        cand = []
        for a in ip:
            for b in ir:
                d = abs((pred[a, 1] + pred[a, 2]) / 2 - (ref[b, 1] + ref[b, 2]) / 2)
                if d <= TOL:
                    cand.append((d, a, b))
        for _, a, b in sorted(cand):
            if a not in tp_p and b not in tp_r:
                tp_p.add(a)
                tp_r.add(b)
                err += [(pred[a, 1] - ref[b, 1]) * 1000 / FS, (pred[a, 2] - ref[b, 2]) * 1000 / FS]
    return tp_p, tp_r, np.abs(err)


def in_zone(pred: np.ndarray, zone) -> np.ndarray:
    """Детекции с центром внутри размеченной зоны (правило краёв M1) — для подсчёта FP."""
    if zone is None:
        return np.ones(len(pred), bool)
    c = (pred[:, 1] + pred[:, 2]) / 2
    return (c >= zone[0]) & (c <= zone[1])


def stats(pred, ref, zone=None) -> dict:
    tp_p, tp_r, err = match(pred, ref)
    inside = in_zone(pred, zone)
    fp = [i for i in range(len(pred)) if i not in tp_p and inside[i]]
    fn = [i for i in range(len(ref)) if i not in tp_r]
    return {"tp": len(tp_r), "fp": len(fp), "fn": len(fn), "fp_idx": fp, "fn_idx": fn,
            "mae": float(err.mean()) if len(err) else float("nan")}


# ---------------------------------------------------------------- рисование (копия plot_predictions.py)
def draw_signal(ax, x, title, zone=None, n_show=None):
    n = x.shape[-1]
    t = np.arange(n) / FS
    ax.plot(t, x, "k", lw=0.7)
    if zone is not None:  # неразмеченные края — серым
        ax.axvspan(0, zone[0] / FS, color="0.85", lw=0, zorder=0)
        ax.axvspan(zone[1] / FS, (n_show or n) / FS, color="0.85", lw=0, zorder=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, (n_show or n) / FS)
    ax.tick_params(labelsize=8)


def draw_intervals(ax, intervals, hatch_idx=(), alpha=0.28):
    lo, hi = ax.get_ylim()
    for i, (c, l, r, *_) in enumerate(intervals):
        bad = i in hatch_idx
        ax.add_patch(Rectangle((l / FS, lo), (r - l) / FS, hi - lo, facecolor=COLORS[int(c)], alpha=alpha,
                               edgecolor="m" if bad else "none", hatch="//" if bad else None, lw=1.5 if bad else 0))


def draw_missed(ax, ref, idx):
    lo, hi = ax.get_ylim()
    for i in idx:
        c, l, r = ref[i]
        ax.add_patch(Rectangle((l / FS, lo), (r - l) / FS, hi - lo, fill=False, edgecolor=COLORS[int(c)],
                               ls="--", lw=1.5))


def panel(ax, x, title_prefix, pred, ref, zone=None, n_show=None):
    """Строка «предсказание модели»: найденные волны — заливка, ложные — штриховка с пурпурной рамкой,
    пропущенные волны эксперта — пунктир."""
    s = stats(pred, ref, zone)
    draw_signal(ax, x, f"{title_prefix}", zone, n_show)
    draw_intervals(ax, pred, set(s["fp_idx"]))
    draw_missed(ax, ref, s["fn_idx"])
    return s


def legend(fig):
    handles = [Patch(color=c, alpha=0.4, label=n) for c, n in zip(COLORS, NAMES)]
    handles += [
        # Patch(facecolor="w", edgecolor="m", hatch="//", label="False Positive"),
                Patch(facecolor="w", edgecolor="k", ls="--", label="False Negative"),
                Patch(color="0.85", label="Unlabeled Part")
                ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12, frameon=False)


def main(SID, axes):
    rec = SID.split("_")[0]
    lead = SID.split("_")[1]
    with open(REPO / "data" / "splits" / "folds.csv", encoding="utf-8") as f:
        k = {r["record"]: int(r["fold"]) for r in csv.DictReader(f)}[rec]
    with open(REPO / "data" / "processed" / "ludb" / "manifest.csv", encoding="utf-8") as f:
        zone = {r["sample_id"]: (int(r["annot_start"]), int(r["annot_end"])) for r in csv.DictReader(f)}.get(SID)
    fold_yaml = REPO / "data" / "splits" / f"fold_{k}.yaml"
    pp.FOLD_YAML = fold_yaml  # Model берёт настройки датасета отсюда
    model = pp.Model(REPO / "experiments" / "final" / f"fold{k}_seed{SEED}")

    ds = Dataset(str(fold_yaml), None, "test", edge_mask=EdgeMask((6, 6), (12, 12)))
    i = [Path(f).stem for f in ds.filenames].index(SID)
    x = ds.load_signal(i)                                        # (1, n), вход как при оценке
    ref = boxes_to_ref(ds.labels[ds.filenames[i]], x.shape[-1])  # (m, 3) [класс, начало, конец], отсчёты
    pred = model.predict(x)                                      # (k, 4) [класс, начало, конец, conf]

    # fig, axes = plt.subplots(2, 1, figsize=(15, 5.2), sharex=True)
    draw_signal(axes[0], x[0].numpy(), f"Record {rec} lead {str.upper(lead)}: Groud Truth", zone)
    draw_intervals(axes[0], np.c_[ref, np.ones(len(ref))])
    panel(axes[1], x[0].numpy(), f"Record {rec} lead {str.upper(lead)}: Predictions", pred, ref, zone)
    if T_RANGE is not None:
        axes[1].set_xlim(*T_RANGE)
    


if __name__ == "__main__":
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.66), sharex=True)
    main("116_v2", axes=axes[2:])
    main("116_ii", axes=axes[:2])
    legend(fig)
    axes[-1].set_xlabel("Time, s", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pic_11_ECG_predictions.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)