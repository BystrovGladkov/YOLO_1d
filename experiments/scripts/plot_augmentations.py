"""Визуальная проверка ЭКГ-аугментаций (A1): по 3 примера на каждую → docs/augmentations/<ID>_<имя>.png.

Каждый пример — отведение LUDB из train фолда 0 (маска краёв как на валидации): сверху исходный
сигнал с разметкой, снизу — после аугментации (p = 1) с пересчитанной разметкой.
Использование: python experiments/scripts/plot_augmentations.py
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402
from yolo_1d.utils.transforms import ecg_transforms as T  # noqa: E402

OUT = REPO / "docs" / "augmentations"
COLORS = ["tab:green", "tab:red", "tab:blue"]
AUGS = {
    "N1_baseline_wander": T.BaselineWander(1.0),
    "N2_powerline": T.PowerlineNoise(1.0),
    "N3_emg_bursts": T.EMGBursts(1.0),
    "N4_white_noise": T.WhiteNoise(1.0),
    "N5_nstdb": T.NSTDBNoise(1.0),
    "N6_baseline_step": T.BaselineStep(1.0),
    "N7_random_filter": T.RandomFilter(1.0),
    "T1_time_scale": T.TimeScale(1.0),
    "T2_hrv_warp": T.HRVWarp(1.0),
    "T3_window_warp": T.WindowWarp(1.0),
    "A1_amplitude_scale": T.AmplitudeScale(1.0),
    "A2_wave_scale": T.WaveScale(1.0),
    "A3_invert": T.Invert(1.0),
    "M1_time_mask": T.TimeMask(1.0),
}
SAMPLES = ["1_ii", "35_v1", "116_v1"]  # синус / ФП / синус с другой морфологией (если в train фолда 0)


def draw(ax, x, boxes, cls, title):
    n = x.shape[-1]
    ax.plot(np.arange(n) / 500, x[0].numpy(), "k", lw=0.6)
    for (c, w), k in zip(boxes.tolist(), cls.reshape(-1).tolist()):
        ax.axvspan((c - w / 2) * n / 500, (c + w / 2) * n / 500, color=COLORS[int(k)], alpha=0.25, lw=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 11)


def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = Dataset(str(REPO / "data" / "splits" / "fold_0.yaml"), None, "train", edge_mask=EdgeMask((6, 6), (12, 12)))
    ids = [Path(f).stem for f in ds.filenames]
    picks = [ids.index(s) for s in SAMPLES if s in ids]
    picks += random.sample(range(len(ids)), 3 - len(picks))

    def sample(i):
        label = ds.labels[ds.filenames[i]]
        return ds.load_signal(i), torch.as_tensor(label[:, 1:3]), torch.as_tensor(label[:, :1])

    for name, op in AUGS.items():
        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
        for row, i in enumerate(picks[:3]):
            x, b, c = sample(i)
            y, b2, c2 = op(x.clone(), b.clone(), c.clone())
            draw(axes[2 * row], x, b, c, f"{ids[i]}: исходный")
            draw(axes[2 * row + 1], y, b2, c2, f"{ids[i]}: {name}")
        fig.suptitle(f"{name} ({type(op).__name__}, p = 1); P — зелёный, QRS — красный, T — синий")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=80)
        plt.close(fig)
        print(OUT / f"{name}.png")

    for name, aligned in (("X1_mosaic", False), ("X2_beat_splice", True)):
        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
        for row, i in enumerate(picks[:3]):
            data, label = ds.load_splice(i, aligned=aligned)
            draw(axes[row], data, torch.as_tensor(label[:, 1:3]), torch.as_tensor(label[:, :1]),
                 f"{ids[i]} + случайная запись: {name}")
        fig.suptitle(f"{name}: склейка двух записей")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=80)
        plt.close(fig)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
