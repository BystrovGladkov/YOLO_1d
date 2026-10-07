"""E1: анализ smoke-прогона — граница маски, train vs test, память и скорость.

1. Граница «нули → сигнал» (R4 §5.4). Если сеть использует границу маскирования как подсказку,
   onset первой волны отведения (граница маски стоит в δ = 20 мс перед ним) и offset последней волны
   будут найдены заметно точнее / чаще, чем те же точки внутренних волн того же класса.
   Сравниваются Se, среднее и SD ошибки: крайние волны vs внутренние, по классам.
2. Train vs test: метрики лучшего чекпоинта на train-сплите фолда (с порогами, подобранными на val)
   против test → переобучение / недообучение.
3. Пиковая память GPU на шаге обучения (batch из args.yaml) и время эпохи.

Использование:
    python experiments/scripts/e1_analysis.py experiments/e1_fold0_short/fold0_seed0
Результат: <run_dir>/analysis.json (+ печать сводки).
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import (DelineationMetrics, apply_thresholds, filter_to_zone,  # noqa: E402
                                       match_points)
from yolo_1d.utils.settings import read_settings  # noqa: E402

TOL = 75  # 150 мс в отсчётах


def load_run(run_dir: Path):
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = {k: (v["conf"], v["val_F1"]) for k, v in thresholds.items()}
    preds = json.loads((run_dir / "predictions_test.json").read_text(encoding="utf-8"))
    return thresholds, preds


def edge_analysis(preds, thresholds, names=("P", "QRS", "T")) -> dict:
    """Точность onset первой волны / offset последней волны отведения против внутренних волн того же класса."""
    stats = {}  # (class, kind) -> {"found": [...], "err": [...]}
    for s in preds:
        pred = apply_thresholds(np.asarray(s["pred"], float).reshape(-1, 4), thresholds, names)
        ref = np.asarray(s["ref"], float).reshape(-1, 3)
        if len(ref) == 0:
            continue
        pred = filter_to_zone(pred[:, :3], ref)
        first, last = int(np.argmin(ref[:, 1])), int(np.argmax(ref[:, 2]))
        for c, name in enumerate(names):
            idx = np.nonzero(ref[:, 0] == c)[0]
            pc = pred[pred[:, 0] == c]
            for col, edge_wave, edge_kind in ((1, first, "onset"), (2, last, "offset")):
                pairs, err = match_points(pc[:, col], ref[idx, col], TOL)
                matched = {int(idx[r]): e for (_, r), e in zip(pairs, err)}
                for k in idx:
                    kind = f"{'first' if edge_kind == 'onset' else 'last'}_{edge_kind}" if k == edge_wave \
                        else f"interior_{edge_kind}"
                    d = stats.setdefault((name, kind), {"found": [], "err": []})
                    d["found"].append(k in matched)
                    if k in matched:
                        d["err"].append(matched[k] * 2.0)  # отсчёты -> мс
    out = {}
    for (name, kind), d in sorted(stats.items()):
        err = np.asarray(d["err"])
        out[f"{name}/{kind}"] = {"n": len(d["found"]), "Se": float(np.mean(d["found"])),
                                 "mean_ms": float(err.mean()) if err.size else None,
                                 "sd_ms": float(err.std()) if err.size else None}
    return out


def train_split_metrics(run_dir: Path, thresholds) -> dict:
    """Метрики лучшего чекпоинта на train-сплите (test-ключи датасета подменяются train-списком)."""
    from yolo_1d.yolo_1d import Trainer

    train_dir = run_dir / json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["train_dir"]
    settings = read_settings(str(train_dir / "args.yaml"))
    dataset = yaml.safe_load(Path(settings.dataset).read_text(encoding="utf-8"))
    base = (Path(settings.dataset).parent / dataset.get("path", ".")).resolve()
    for key in ("data", "labels", "list"):
        dataset[f"test_{key}"] = dataset[f"train_{key}"]
    for split in ("train", "val", "test"):
        for key in ("data", "labels", "list"):
            dataset[f"{split}_{key}"] = str(base / dataset[f"{split}_{key}"])
    dataset.pop("path", None)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "train_as_test.yaml"
        cfg.write_text(yaml.safe_dump(dataset), encoding="utf-8")
        settings.dataset = str(cfg)
        settings.results_path = tmp
        trainer = Trainer(settings)
        trainer.load_checkpoint(str(train_dir / "best.pt"))
        samples, _ = trainer.collect_predictions("test")
    metrics = DelineationMetrics(class_names=settings.names)
    for s in samples:
        metrics.add(apply_thresholds(s["pred"], thresholds, settings.names), s["ref"])
    return metrics.compute()


def memory_and_speed(run_dir: Path) -> dict:
    """Пиковая память GPU на шаге обучения и среднее время эпохи."""
    from yolo_1d.nets import nn
    from yolo_1d.nets.criterion import ComputeLoss

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    train_dir = run_dir / run["train_dir"]
    settings = read_settings(str(train_dir / "args.yaml"))
    epochs = len(pd.read_csv(train_dir / "step.csv"))
    out = {"epochs": epochs, "run_duration_s": run["duration_s"],
           "seconds_per_epoch_incl_val": round(run["duration_s"] / epochs, 2)}
    if torch.cuda.is_available():
        aug = settings.augmentations
        model = nn.get_model(settings.model_size, len(settings.names), settings.input_channels, aug.input_length,
                             strides=settings.backbone_strides, bias_init=settings.bias_init).cuda().train()
        criterion = ComputeLoss(model, settings.weight_box, settings.weight_cls, settings.weight_dfl)
        x = torch.randn(settings.batch_size, settings.input_channels, aug.input_length, device="cuda")
        n = 20 * settings.batch_size
        targets = {"cls": torch.randint(0, 3, (n, 1)).float(), "box": torch.rand(n, 2) * 0.02 + torch.tensor([0.4, 0.01]),
                   "idx": torch.arange(settings.batch_size).repeat_interleave(20).float()}
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(5):
            with torch.amp.autocast("cuda"):
                loss = sum(criterion(model(x), targets))
            loss.backward()
        torch.cuda.synchronize()
        out["gpu_peak_memory_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out["train_step_ms"] = round((time.time() - start) / 5 * 1000, 1)
    return out


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    thresholds, preds = load_run(run_dir)
    result = {"edges": edge_analysis(preds, thresholds), "memory_speed": memory_and_speed(run_dir),
              "test": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["test"]["summary"]}
    if "--train" in sys.argv:
        train = train_split_metrics(run_dir, thresholds)
        result["train"] = {"summary": train["summary"], "points": {k: v["F1"] for k, v in train["points"].items()}}
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "edges"}, indent=2, ensure_ascii=False))
    print("\nГраница маски: крайние волны vs внутренние")
    for key, v in result["edges"].items():
        m = "—" if v["mean_ms"] is None else f"{v['mean_ms']:6.1f} ± {v['sd_ms']:5.1f} мс"
        print(f"  {key:22s} n={v['n']:5d}  Se={100 * v['Se']:6.2f} %  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
