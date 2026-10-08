"""Демонстрация работы сети на валидации фолда 0: разметка эксперта против предсказаний Full (все аугментации)
и baseline (без аугментаций). Картинки — docs/results/e3_examples/.

1. `val_<запись>_<диагноз>.png` — записи с патологиями из val фолда 0, вход как при оценке (маска краёв).
   Для каждой записи — отведение II и отведение, где модели расходятся сильнее всего.
2. `heavy_aug_<k>.png` — на сигнал val наложено сразу много аугментаций (время + амплитуда + шумы).
3. `unmasked_<запись>.png` — полная запись без маскирования неразмеченных краёв.
Плюс `val_errors.md` — счётчики TP/FP/FN по всем отведениям val для обеих моделей.

Сопоставление на картинках упрощённое (для наглядности): волна найдена, если центр предсказания одного
класса ближе 150 мс к центру волны эксперта (один-к-одному). Метрики статьи считаются по 6 точкам (M1).
    python experiments/scripts/plot_predictions.py
"""
import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import util  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.delineation import apply_thresholds, detections_to_intervals  # noqa: E402
from yolo_1d.utils.ludb import load_metadata  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402
from yolo_1d.utils.transforms import ecg_transforms as T  # noqa: E402

OUT = REPO / "docs" / "results" / "e3_examples"
FOLD_YAML = REPO / "data" / "splits" / "fold_0.yaml"
MODELS = {"Full (все аугментации)": REPO / "experiments" / "abl_full" / "fold0_seed0",
          "baseline (без аугментаций)": REPO / "experiments" / "baseline" / "fold0_seed0"}
FS, INPUT, TOL = 500, 5120, 75  # Гц, длина входа, допуск сопоставления 150 мс в отсчётах
NAMES = ["P", "QRS", "T"]
COLORS = ["tab:green", "tab:red", "tab:blue"]
# записи val фолда 0 с патологиями → короткая подпись
RECORDS = {
    "35": "трепетание предсердий",
    "111": "ЭКС, AV-блокада III ст.",
    "132": "предсердная бигеминия",
    "7": "предсердная экстрасистолия",
    "33": "STEMI (передне-боковой)",
    "54": "неполная БЛНПГ, ГЛЖ",
    "70": "неполная БПНПГ, синусовая тахикардия",
    "151": "синусовый ритм, ГЛЖ (почти норма)",
}
HEAVY = [("70_ii", 3), ("132_v2", 5), ("6_v5", 7)]  # (отведение, сид) для примеров с многими аугментациями
UNMASKED = ["151_ii", "35_ii", "111_v1"]


# ---------------------------------------------------------------- модели и предсказания
class Model:
    """EMA-модель из best.pt прогона + пороги уверенности по классам с val этого прогона."""

    def __init__(self, run_dir: Path):
        from yolo_1d.yolo_1d import Trainer
        config = yaml.safe_load((run_dir.parent / "config.yaml").read_text(encoding="utf-8"))
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self._tmp = tempfile.TemporaryDirectory()
        settings = TrainingSettings(**config["training"], dataset=str(FOLD_YAML), results_path=self._tmp.name,
                                    project="plot", seed=0)
        trainer = Trainer(settings)
        trainer.load_checkpoint(str(run_dir / info["train_dir"] / "best.pt"))
        self.net = trainer._ema.ema.eval()
        self.device = trainer._device
        self.nms_iou = settings.nms_iou
        th = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = {k: (v["conf"], v["val_F1"]) for k, v in th.items()}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """x (1, n), n ≤ 5120 → (k, 4) [класс, начало, конец, conf] в отсчётах x (нули справа, как на валидации)."""
        n = x.shape[-1]
        inp = torch.zeros(1, 1, INPUT)
        inp[0, :, :n] = x
        out = self.net(inp.to(self.device))
        det = util.non_max_suppression(out, 0.001, self.nms_iou)[0].cpu().numpy()
        return apply_thresholds(detections_to_intervals(det, signal_length=n), self.thresholds, NAMES)


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


# ---------------------------------------------------------------- рисование
def draw_signal(ax, x, title, zone=None, n_show=None):
    n = x.shape[-1]
    t = np.arange(n) / FS
    ax.plot(t, x, "k", lw=0.7)
    if zone is not None:  # неразмеченные края — серым
        ax.axvspan(0, zone[0] / FS, color="0.85", lw=0, zorder=0)
        ax.axvspan(zone[1] / FS, (n_show or n) / FS, color="0.85", lw=0, zorder=0)
    ax.set_title(title, fontsize=9, loc="left")
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
        ax.text((l + r) / 2 / FS, hi, "пропуск", ha="center", va="top", fontsize=7, color=COLORS[int(c)])


def panel(ax, x, title_prefix, pred, ref, zone=None, n_show=None):
    """Строка «предсказание модели»: найденные волны — заливка, ложные — штриховка с пурпурной рамкой,
    пропущенные волны эксперта — пунктир."""
    s = stats(pred, ref, zone)
    draw_signal(ax, x, f"{title_prefix}: найдено {s['tp']}, ложных {s['fp']}, пропущено {s['fn']}, "
                       f"ошибка границ |Δ| = {s['mae']:.1f} мс", zone, n_show)
    draw_intervals(ax, pred, set(s["fp_idx"]))
    draw_missed(ax, ref, s["fn_idx"])
    return s


def legend(fig):
    handles = [Patch(color=c, alpha=0.4, label=n) for c, n in zip(COLORS, NAMES)]
    handles += [Patch(facecolor="w", edgecolor="m", hatch="//", label="ложная детекция"),
                Patch(facecolor="w", edgecolor="k", ls="--", label="пропуск (есть у эксперта)"),
                Patch(color="0.85", label="нет разметки эксперта")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)


# ---------------------------------------------------------------- 1. записи с патологиями
def plot_records(models, ds, ids, zones, errors_by_lead):
    for rec, diag in RECORDS.items():
        leads = [i for i in ids if i.split("_")[0] == rec]
        if not leads:
            continue
        # отведение II + отведение с наибольшим расхождением моделей (по числу ошибок, затем по |Δ|)
        def gap(i):
            a, b = (errors_by_lead[i][m] for m in models)
            return (abs((a["fp"] + a["fn"]) - (b["fp"] + b["fn"])), abs(a["mae"] - b["mae"]))
        chosen = [rec + "_ii"] if rec + "_ii" in leads else []
        chosen += [max((i for i in leads if i not in chosen), key=gap)]
        fig, axes = plt.subplots(3 * len(chosen), 1, figsize=(15, 2.0 * 3 * len(chosen) + 0.6), sharex=True)
        for k, sid in enumerate(chosen):
            i = ids.index(sid)
            x = ds.load_signal(i)[0].numpy()
            ref = boxes_to_ref(ds.labels[ds.filenames[i]], len(x))
            zone = zones.get(sid)
            ax = axes[3 * k]
            draw_signal(ax, x, f"{sid}: разметка эксперта (ожидаемый ответ)", zone)
            draw_intervals(ax, np.c_[ref, np.ones(len(ref))])
            for j, name in enumerate(models):
                panel(axes[3 * k + 1 + j], x, f"{sid}: {name}", errors_by_lead[sid][name]["pred"], ref, zone)
        axes[-1].set_xlabel("время, с")
        fig.suptitle(f"Запись {rec} (val фолда 0): {diag}", fontsize=11)
        legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = OUT / f"val_{rec}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


# ---------------------------------------------------------------- 2. много аугментаций сразу
def heavy_pipeline():
    """Тяжёлая комбинация: время + амплитуда + шумы + фильтрация (все p = 1, кроме инверсии и маски)."""
    return [T.HRVWarp(1.0), T.TimeScale(1.0), T.WaveScale(1.0), T.AmplitudeScale(1.0), T.Invert(0.5),
            T.BaselineWander(1.0), T.BaselineStep(1.0), T.NSTDBNoise(1.0), T.EMGBursts(1.0),
            T.PowerlineNoise(1.0), T.WhiteNoise(1.0), T.RandomFilter(1.0), T.TimeMask(0.5),
            T.FitToLength(INPUT, random_shift=False)]  # как в обучении: длина входа 5120


def plot_heavy(models, ds, ids):
    for k, (sid, seed) in enumerate(HEAVY, 1):
        if sid not in ids:
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        i = ids.index(sid)
        x0 = ds.load_signal(i)
        label = ds.labels[ds.filenames[i]]
        box, cls = torch.as_tensor(label[:, 1:3]), torch.as_tensor(label[:, :1])
        x, b, c = x0.clone(), box.clone(), cls.clone()
        applied = []
        for op in heavy_pipeline():
            before = x.clone()
            x, b, c = op(x, b, c)
            if not isinstance(op, T.FitToLength) and (before.shape != x.shape or not torch.equal(before, x)):
                applied.append(type(op).__name__)
        ref0 = boxes_to_ref(label, x0.shape[-1])
        ref = boxes_to_ref(np.c_[c.numpy(), b.numpy()], x.shape[-1])
        n_show = max(x0.shape[-1], x.shape[-1])
        fig, axes = plt.subplots(5, 1, figsize=(15, 11), sharex=True)
        first = list(models)[0]
        draw_signal(axes[0], x0[0].numpy(), f"{sid}: исходный сигнал и разметка эксперта", n_show=n_show)
        draw_intervals(axes[0], np.c_[ref0, np.ones(len(ref0))])
        s = panel(axes[1], x0[0].numpy(), f"исходный, {first}", models[first].predict(x0), ref0, n_show=n_show)
        draw_signal(axes[2], x[0].numpy(), "после аугментаций: пересчитанная разметка (ожидаемый ответ)", n_show=n_show)
        draw_intervals(axes[2], np.c_[ref, np.ones(len(ref))])
        for j, (name, m) in enumerate(models.items()):
            panel(axes[3 + j], x[0].numpy(), f"после аугментаций, {name}", m.predict(x), ref, n_show=n_show)
        axes[-1].set_xlabel("время, с")
        fig.suptitle(f"Много аугментаций сразу ({sid}): " + ", ".join(applied), fontsize=10)
        legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        path = OUT / f"heavy_aug_{k}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


# ---------------------------------------------------------------- 3. без маскирования краёв
def plot_unmasked(models, ds_raw, ids, zones):
    for sid in UNMASKED:
        if sid not in ids:
            continue
        i = ids.index(sid)
        x = ds_raw.load_signal(i)  # маска краёв не применяется
        ref = boxes_to_ref(ds_raw.labels[ds_raw.filenames[i]], x.shape[-1])
        zone = zones.get(sid)
        fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
        draw_signal(axes[0], x[0].numpy(), f"{sid}: полный сигнал без маски, разметка эксперта", zone)
        draw_intervals(axes[0], np.c_[ref, np.ones(len(ref))])
        for j, (name, m) in enumerate(models.items()):
            pred = m.predict(x)
            s = panel(axes[1 + j], x[0].numpy(), name, pred, ref, zone)
            outside = (~in_zone(pred, zone)).sum()
            axes[1 + j].set_title(axes[1 + j].get_title(loc="left") + f"; в неразмеченных краях найдено волн: {outside}",
                                  fontsize=9, loc="left")
        axes[-1].set_xlabel("время, с")
        fig.suptitle(f"{sid}: вход без маскирования неразмеченных краёв (серое — эксперт не размечал, "
                     "ложные там не считаются)", fontsize=10)
        legend(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = OUT / f"unmasked_{sid}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {name: Model(path) for name, path in MODELS.items()}
    ds = Dataset(str(FOLD_YAML), None, "val", edge_mask=EdgeMask((6, 6), (12, 12)))
    ds_raw = Dataset(str(FOLD_YAML), None, "val")
    ids = [Path(f).stem for f in ds.filenames]
    with open(REPO / "data" / "processed" / "ludb" / "manifest.csv", encoding="utf-8") as f:
        zones = {r["sample_id"]: (int(r["annot_start"]), int(r["annot_end"])) for r in csv.DictReader(f)}

    # предсказания на всех отведениях val (вход как при оценке)
    errors = {}
    for i, sid in enumerate(ids):
        x = ds.load_signal(i)
        ref = boxes_to_ref(ds.labels[ds.filenames[i]], x.shape[-1])
        errors[sid] = {}
        for name, m in models.items():
            pred = m.predict(x)
            errors[sid][name] = {**stats(pred, ref, zones.get(sid)), "pred": pred}

    meta = load_metadata(REPO / "data" / "raw" / "ludb")
    lines = ["# Ошибки на val фолда 0 по записям (упрощённое сопоставление волн, ≤ 150 мс по центру)", "",
             "| Запись | Ритм | " + " | ".join(f"{n}: FP / FN / ⟨|Δ|⟩, мс" for n in models) + " |",
             "|---|---|" + "---|" * len(models)]
    for rec in sorted({s.split("_")[0] for s in ids}, key=int):
        leads = [s for s in ids if s.split("_")[0] == rec]
        cells = []
        for name in models:
            e = [errors[s][name] for s in leads]
            cells.append(f"{sum(v['fp'] for v in e)} / {sum(v['fn'] for v in e)} / "
                         f"{np.nanmean([v['mae'] for v in e]):.1f}")
        lines.append(f"| {rec} | {', '.join(meta[rec]['Rhythms'])} | " + " | ".join(cells) + " |")
    (OUT / "val_errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    plot_records(models, ds, ids, zones, errors)
    plot_heavy(models, ds, ids)
    plot_unmasked(models, ds_raw, ids, zones)


if __name__ == "__main__":
    sys.exit(main())
