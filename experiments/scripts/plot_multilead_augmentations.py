"""Визуальная проверка многоканальных аугментаций режима MC (A3) → docs/augmentations/multilead/<ID>.png.

Запись из train фолда 0 (маска краёв как на валидации): слева 12 отведений до аугментации, справа — после
(p = 1), разметка каждого отведения — цветом (P — зелёный, QRS — красный, T — синий).
    python experiments/scripts/plot_multilead_augmentations.py
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
from yolo_1d.utils.ludb import LEADS  # noqa: E402
from yolo_1d.utils.multilead_dataset import MultiLeadDataset  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask, build_ecg_multilead_training_transforms  # noqa: E402
from yolo_1d.utils.transforms import ecg_multilead_transforms as M  # noqa: E402
from yolo_1d.utils.transforms import ecg_transforms as T  # noqa: E402

OUT = REPO / "docs" / "augmentations" / "multilead"
COLORS = ["tab:green", "tab:red", "tab:blue"]
RECORD = "1"

SIGNAL_AUGS = {  # ID → (подпись, аугментация с p = 1)
    "T2_hrv_warp": ("T2 HRV-варп: одно отображение времени по кластерам волн всех отведений", M.MultiLeadHRVWarp(1.0)),
    "A1_amplitude": ("A1 общий × поканальный масштаб амплитуды", M.MultiLeadAmplitudeScale(1.0, 0.2, 0.1)),
    "A2_wave_scale": ("A2 масштаб волн: общий множитель волны × поканальный", M.MultiLeadWaveScale(1.0)),
    "N1_wander": ("N1 дрейф изолинии: общий источник, коэффициент отведения U(0.5, 1.5)·(±1)",
                  M.KeepZeros(M.CoupledNoise(1.0, "wander"))),
    "N5_nstdb": ("N5 шум NSTDB: общий фрагмент, коэффициент отведения", M.KeepZeros(M.CoupledNoise(1.0, "nstdb", snr=(6, 12)))),
    "N6_step": ("N6 ступенька изолинии в 1–3 отведениях", M.KeepZeros(M.LeadBaselineStep(1.0))),
    "M1_time_mask": ("M1 обнуление отрезка — общий для всех отведений", T.TimeMask(1.0, max_fraction=0.15)),
}
LEAD_AUGS = {
    "L1_lead_dropout": ("L1 lead dropout (разметка обнулённых отведений игнорируется)",
                        M.LeadDropout(1.0, p_lead=0.25, max_leads=4)),
    "L2_lead_shuffle": ("L2 перестановка отведений вместе с разметкой", M.LeadShuffle(1.0)),
    "L3_electrode_LA_RA": ("L3 перестановка электродов LA↔RA: I → −I, II ↔ III, aVR ↔ aVL", None),
}


def draw(ax, x, boxes, labels, title, valid=None):
    n = x.shape[-1]
    t = np.arange(n) / 500
    for k in range(x.shape[0]):
        off = -4.0 * k
        ax.plot(t, x[k].numpy() + off, "k" if valid is None or valid[k] else "0.6", lw=0.5)
    left, right = T.boxes_to_edges(boxes, n)
    for l, r, (c, k) in zip(left, right, labels.long().tolist()):
        ax.fill_between([l / 500, r / 500], -4.0 * k - 1.8, -4.0 * k + 1.8, color=COLORS[c], alpha=0.3, lw=0)
    ax.set_yticks([-4.0 * k for k in range(x.shape[0])], [l.upper() for l in LEADS], fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, n / 500)


def sample(ds, i):
    label = ds.labels[ds.filenames[i]]
    return ds.load_signal(i), torch.from_numpy(label[:, 2:4].copy()), torch.from_numpy(label[:, 0:2].copy()), ds.valid(i)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=70)
    plt.close(fig)
    print(OUT / f"{name}.png")


def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = MultiLeadDataset(str(REPO / "data" / "splits" / "fold_0_mc.yaml"), None, "train",
                          edge_mask=EdgeMask((6, 6), (12, 12)))
    ids = [ds.sample_id(i) for i in range(ds.n)]
    i = ids.index(RECORD) if RECORD in ids else 0
    x, b, lab, valid = sample(ds, i)

    def pair(name, title, y, b2, lab2, v2=None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        draw(axes[0], x, b, lab, f"запись {ids[i]}: исходная", valid)
        draw(axes[1], y, b2, lab2, title, valid if v2 is None else v2)
        save(fig, name)

    for name, (title, op) in SIGNAL_AUGS.items():
        y, b2, lab2 = op(x.clone(), b.clone(), lab.clone())
        pair(name, title, y, b2, lab2)
    for name, (title, op) in LEAD_AUGS.items():
        if op is None:
            y, b2, lab2, v2 = M.ElectrodeSwap.swap("LA-RA", x.clone(), b.clone(), lab.clone(), valid.clone())
        else:
            y, b2, lab2, v2 = op(x.clone(), b.clone(), lab.clone(), valid.clone())
        pair(name, title + " (серым — отведения вне лосса)", y, b2, lab2, v2)

    # X2: склейка по паузам, общий разрез для всех отведений
    ds.beat_splice_prob = 1.0
    y, label, v2 = ds.load_splice(i, aligned=True)
    pair("X2_beat_splice", "X2 склейка двух записей: общий разрез в паузе после T во всех отведениях", y,
         torch.from_numpy(label[:, 2:4].copy()), torch.from_numpy(label[:, 0:2].copy()), v2)

    # полный пайплайн (вероятности final.yaml, без A3) — 2 примера
    import yaml
    config = yaml.safe_load((REPO / "experiments" / "configs" / "final.yaml").read_text(encoding="utf-8"))
    aug = {**config["training"]["augmentations"], "pipeline": "ecg_multi"}
    pipeline = build_ecg_multilead_training_transforms(AugmentationSettings(**aug))
    for k in (1, 2):
        y, b2, lab2 = pipeline(x.clone(), b.clone(), lab.clone())
        pair(f"full_pipeline_{k}", "все сигнальные аугментации MC (вероятности E4) + приведение к 5120", y, b2, lab2)


if __name__ == "__main__":
    sys.exit(main())
