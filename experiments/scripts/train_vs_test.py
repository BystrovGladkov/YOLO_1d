"""Переобучение: метрики делинеации лучшего чекпоинта на train фолда против test.

Для каждого прогона `<exp>/fold{k}_seed{s}`: загружается best.pt, train-список фолда подаётся как
test-сплит (без аугментаций, маска краёв как на eval), детекции фильтруются порогами из
thresholds.json (подобраны на val), метрики сравниваются с test из metrics.json.

    python experiments/scripts/train_vs_test.py experiments/abl_full experiments/baseline --runs fold0_seed0 fold3_seed0
Вывод — Markdown-таблица в консоль и `<exp>/train_vs_test.json`.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import DelineationMetrics, apply_thresholds  # noqa: E402
from yolo_1d.utils.settings import TrainingSettings  # noqa: E402


def train_metrics(exp_dir: Path, run: str) -> dict:
    """Метрики лучшего чекпоинта прогона на train-списке его фолда."""
    from yolo_1d.yolo_1d import Trainer

    config = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    info = json.loads((exp_dir / run / "run.json").read_text(encoding="utf-8"))
    fold_yaml = REPO / config["dataset"].format(fold=info["fold"])
    data = yaml.safe_load(fold_yaml.read_text(encoding="utf-8"))
    data["path"] = str((fold_yaml.parent / data["path"]).resolve())
    for key in ("data", "labels", "list"):
        data[f"test_{key}"] = data[f"train_{key}"]
    thresholds = json.loads((exp_dir / run / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = {k: (v["conf"], v["val_F1"]) for k, v in thresholds.items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_yaml = Path(tmp) / "train_as_test.yaml"
        tmp_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        settings = TrainingSettings(**config["training"], dataset=str(tmp_yaml), results_path=tmp,
                                    project="eval", seed=info["seed"])
        trainer = Trainer(settings)
        trainer.load_checkpoint(str(exp_dir / run / info["train_dir"] / "best.pt"))
        samples, _ = trainer.collect_predictions("test")
        metrics = DelineationMetrics(class_names=settings.names,
                                     exclude_boundary_points=settings.exclude_boundary_points)
        for s in samples:
            metrics.add(apply_thresholds(s["pred"], thresholds, settings.names), s["ref"])
        del trainer
        torch.cuda.empty_cache()
    return metrics.compute()["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--runs", nargs="+", default=None, help="имена прогонов (по умолчанию все)")
    args = parser.parse_args()
    print("| Эксперимент | Прогон | train F1₆ | test F1₆ | train σ₆ | test σ₆ |\n|---|---|---|---|---|---|")
    for exp_dir in args.experiments:
        runs = args.runs or sorted(p.name for p in exp_dir.glob("fold*_seed*") if (p / "metrics.json").is_file())
        out = {}
        for run in runs:
            test = json.loads((exp_dir / run / "metrics.json").read_text(encoding="utf-8"))["test"]["summary"]
            train = train_metrics(exp_dir, run)
            out[run] = {"train": train, "test": test}
            print(f"| {exp_dir.name} | {run} | {100 * train['F1_mean6']:.2f} % | {100 * test['F1_mean6']:.2f} % "
                  f"| {train['sd_mean6_ms']:.1f} мс | {test['sd_mean6_ms']:.1f} мс |", flush=True)
        (exp_dir / "train_vs_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
