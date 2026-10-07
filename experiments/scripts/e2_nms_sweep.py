"""E2: подбор IoU-порога NMS по val (без переобучения).

Для каждого прогона загружается best.pt, val прогоняется с разными `nms_iou`, пороги уверенности
по классам подбираются на том же val (как в итоговой оценке), сравнивается ⟨F1⟩₆ и ⟨σ⟩₆ на val.
Выбор делается только по val; test не используется.

Использование:
    python experiments/scripts/e2_nms_sweep.py experiments/e2s1_A_nano/fold0_seed0 [run_dir ...] \
        [--ious 0.3 0.45 0.65 0.8]
Результат: <run_dir>/nms_sweep.json и сводка в консоль (среднее по прогонам).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import DelineationMetrics, apply_thresholds, best_class_thresholds  # noqa: E402
from yolo_1d.utils.settings import read_settings  # noqa: E402


def sweep(run_dir: Path, ious) -> dict:
    from yolo_1d.yolo_1d import Trainer

    train_dir = run_dir / json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["train_dir"]
    settings = read_settings(str(train_dir / "args.yaml"))
    settings.results_path = str(run_dir / "_sweep_tmp")
    trainer = Trainer(settings)
    trainer.load_checkpoint(str(train_dir / "best.pt"))
    out = {}
    for iou in ious:
        trainer._settings.nms_iou = iou
        val, _ = trainer.collect_predictions("val")
        pairs = [(s["pred"], s["ref"]) for s in val]
        thr = best_class_thresholds(pairs, class_names=settings.names,
                                    exclude_boundary_points=settings.exclude_boundary_points)
        metrics = DelineationMetrics(class_names=settings.names, exclude_boundary_points=settings.exclude_boundary_points)
        for pred, ref in pairs:
            metrics.add(apply_thresholds(pred, thr, settings.names), ref)
        out[str(iou)] = metrics.compute()["summary"]
    (run_dir / "nms_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--ious", nargs="+", type=float, default=[0.3, 0.45, 0.65, 0.8])
    args = parser.parse_args()
    results = {str(r): sweep(r.resolve(), args.ious) for r in args.runs}
    print("IoU   | val F1_mean6 (среднее по прогонам) | val sd_mean6, мс")
    for iou in map(str, args.ious):
        f1 = np.mean([res[iou]["F1_mean6"] for res in results.values()])
        sd = np.mean([res[iou]["sd_mean6_ms"] for res in results.values()])
        print(f"{iou:5s} | {100 * f1:.2f} % | {sd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
