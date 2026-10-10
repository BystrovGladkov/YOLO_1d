"""E7: сравнение режима S (E4 `final`) и MC (`mc_full`) на одних и тех же отведениях test → docs/results/e7/s_vs_mc_<запись>.png.

Для записи берутся модели того фолда, в test которого она попала (seed 2 у обеих моделей), вход как при оценке
(маска краёв 6/12). Строки: разметка эксперта, ответ S, ответ MC (обозначения — plot_predictions.py).
    python experiments/scripts/plot_s_vs_mc.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_predictions as pp  # noqa: E402
from plot_predictions import REPO  # noqa: E402
from yolo_1d.utils import util  # noqa: E402
from yolo_1d.utils.delineation import apply_thresholds, detections_to_intervals  # noqa: E402
from yolo_1d.utils.ludb import LEADS  # noqa: E402
from yolo_1d.utils.multilead_dataset import MultiLeadDataset  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask  # noqa: E402

OUT = REPO / "docs" / "results" / "e7"
SEED = 2
CASES = {
    "116": (["ii", "v2", "avl"], "полная блокада ножки: P почти не видна в V2/aVL (ошибка типа 1 из E4)"),
    "24": (["ii", "v2"], "полная блокада ножки, ишемия: пропуски P в S"),
    "72": (["ii", "v1"], "ЭКС: MC теряет P"),
    "103": (["ii", "v1"], "трепетание предсердий"),
}


class MCModel:
    """EMA-модель режима MC из best.pt + пороги по классам; predict(x (12, n)) → список (12) детекций."""

    def __init__(self, run_dir: Path, fold_yaml: Path):
        from yolo_1d.utils.settings import TrainingSettings
        from yolo_1d.yolo_1d import Trainer
        config = yaml.safe_load((run_dir.parent / "config.yaml").read_text(encoding="utf-8"))
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self._tmp = tempfile.TemporaryDirectory()
        settings = TrainingSettings(**config["training"], dataset=str(fold_yaml), results_path=self._tmp.name,
                                    project="plot", seed=0)
        trainer = Trainer(settings)
        trainer.load_checkpoint(str(run_dir / info["train_dir"] / "best.pt"))
        self.net, self.device, self.nms_iou = trainer._ema.ema.eval(), trainer._device, settings.nms_iou
        th = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = {k: (v["conf"], v["val_F1"]) for k, v in th.items()}

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        n = x.shape[-1]
        inp = torch.zeros(1, x.shape[0], pp.INPUT)
        inp[0, :, :n] = x
        dets = util.non_max_suppression(self.net(inp.to(self.device)), 0.001, self.nms_iou)
        return [apply_thresholds(detections_to_intervals(d.cpu().numpy(), signal_length=n), self.thresholds, pp.NAMES)
                for d in dets]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(REPO / "data" / "splits" / "folds.csv", encoding="utf-8") as f:
        fold_of = {r["record"]: int(r["fold"]) for r in csv.DictReader(f)}
    with open(REPO / "data" / "processed" / "ludb" / "manifest.csv", encoding="utf-8") as f:
        zones = {r["sample_id"]: (int(r["annot_start"]), int(r["annot_end"])) for r in csv.DictReader(f)}
    for rec, (leads, caption) in CASES.items():
        k = fold_of[rec]
        pp.FOLD_YAML = REPO / "data" / "splits" / f"fold_{k}.yaml"
        s_model = pp.Model(REPO / "experiments" / "final" / f"fold{k}_seed{SEED}")
        mc_yaml = REPO / "data" / "splits" / f"fold_{k}_mc.yaml"
        mc_model = MCModel(REPO / "experiments" / "mc_full" / f"fold{k}_seed{SEED}", mc_yaml)
        ds = MultiLeadDataset(str(mc_yaml), None, "test", edge_mask=EdgeMask((6, 6), (12, 12)))
        i = [ds.sample_id(j) for j in range(ds.n)].index(rec)
        x = ds.load_signal(i)
        mc_pred = mc_model.predict(x)
        fig, axes = plt.subplots(3 * len(leads), 1, figsize=(15, 2.0 * 3 * len(leads) + 0.8), sharex=True)
        for j, lead in enumerate(leads):
            li = LEADS.index(lead)
            sid = f"{rec}_{lead}"
            sig = x[li].numpy()
            ref = np.c_[ds.lead_labels(i, li)[:, 0], (ds.lead_labels(i, li)[:, 1] - ds.lead_labels(i, li)[:, 2] / 2) * 5000,
                        (ds.lead_labels(i, li)[:, 1] + ds.lead_labels(i, li)[:, 2] / 2) * 5000]
            zone = zones.get(sid)
            pp.draw_signal(axes[3 * j], sig, f"{sid}: разметка эксперта", zone)
            pp.draw_intervals(axes[3 * j], np.c_[ref, np.ones(len(ref))])
            pp.panel(axes[3 * j + 1], sig, f"{sid}: S (E4, 1 отведение на вход)", s_model.predict(x[li:li + 1]), ref, zone)
            pp.panel(axes[3 * j + 2], sig, f"{sid}: MC (E7, 12 отведений на вход)", mc_pred[li], ref, zone)
        axes[-1].set_xlabel("время, с")
        fig.suptitle(f"Запись {rec} (test фолда {k}, seed {SEED}): {caption}", fontsize=10)
        pp.legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = OUT / f"s_vs_mc_{rec}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    sys.exit(main())
