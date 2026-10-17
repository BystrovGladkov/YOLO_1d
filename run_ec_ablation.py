"""Абляция аугментаций на ECRv1 (вихретоковые дефектограммы): конфиги, обучение, сводка (TASKS.md, D7).

Использование:
    python run_ec_ablation.py make-configs                    # experiments/ec_ablation/configs/{A0..A8,IN_*}.yaml
    python run_ec_ablation.py make-configs --in-base A6       # база для IN_raw / IN_filtered (после E4)
    python run_ec_ablation.py train A3 --seeds 0 1 2          # обучение + оценка val/test
    python run_ec_ablation.py train A0 --seeds 0 --epochs 2   # смоук (E1)
    python run_ec_ablation.py noisy-test A0 A8 --seeds 0 1 2  # E5: оценка на test + SquareNoises
    python run_ec_ablation.py summary                         # summary.csv, summary_agg.csv, summary.md

Конфиги: кумулятивные A0…A8 (таблица `GROUPS`; источник истины — этот файл, yaml генерируются из него) и
IN_raw / IN_filtered (итоговый набор с подмножеством входных каналов, E6; поле `input_subset` — D8). Каждый
yaml — полный и явный: `name`, `description`, `dataset`, `seeds`, `training` (поля TrainingSettings без
`dataset`, `results_path`, `project`, `seed`). Неизвестные поля — ошибка (pydantic их молча игнорирует).

Результаты прогона `experiments/ec_ablation/runs/<ID>_s<seed>/`:
    train1/              step.csv, args.yaml, best.pt, last.pt (Trainer)
    config.yaml          конфиг прогона; environment.json — git hash, версии, GPU, команда
    metrics.json         val/test: mAP@0.5, mAP@0.5:0.95, P, R + AP@0.5, AP@0.5:0.95, P, R по классам
    predictions_test.npz детекции и разметка каждого тестового сигнала (офлайн-bootstrap, R1 §3)
    noisy_test.json      (E5) метрики на test + SquareNoises
    run.json             чекпоинт, длительность
Готовый прогон (есть metrics.json) повторно не обучается; незавершённый — продолжается (`resume`); --rerun — заново.
Seed задаёт инициализацию весов (Trainer вызывает `setup_seed` до создания модели), порядок батчей и аугментации.
"""
import argparse
import copy
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from run_experiment import environment  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402

EXP_DIR = REPO / "experiments" / "ec_ablation"
CONFIG_DIR = EXP_DIR / "configs"
RUNS_DIR = EXP_DIR / "runs"
DATASET = "data/ecr_v1.yaml"
SEEDS = [0, 1, 2]
NAMES = ["flashbutt", "aluminothermic", "rail end", "axle counter", "crossing", "manual arc"]

# базовые настройки обучения (start_example.py); epochs уточняется в E1
BASE_TRAINING = {
    "names": NAMES,
    "model_size": "large",
    "batch_size": 50,
    "epochs": 300,
    "enable_amp": True,
    "early_stopping_epochs": 20,
    "weight_box": 10,
    "weight_dfl": 3,
    "num_workers": 4,
    # выбор best.pt — по mAP@0.5:0.95 на val (решение пользователя 2026-10-17; Trainer: r["mAP"])
    "best_metric": "mAP",
}
# A0: все аугментации выключены, длина приводится как на валидации
AUG_OFF = {
    "pipeline": "eddy", "regular_length": 50000, "input_length": 49152, "validation_resize": True,
    "random_resize": 0, "random_crop": False, "random_reverse": 0, "phase_spectrum_noise": 0, "random_jitter": 0,
    "random_invert": 0, "channel_scaling": 0, "time_masking": 0, "channel_masking": 0, "mosaic": 0, "mix_up": 0,
    "sine_noise": 0, "square_noises": 0, "adversarial": 0, "augment_apply_p": 0.5,
}
# кумулятивные группы (порядок SPEC); каждая аугментация, кроме RandomResize/RandomCrop/RandomPadding, — под
# RandomApply(augment_apply_p × собственная вероятность)
GROUPS = [
    ("A0", "без аугментаций", {}),
    ("A1", "+ геометрия (RandomResize, RandomCrop, RandomPadding, RandomReverse)",
     {"random_resize": 0.1, "random_crop": True, "random_reverse": 0.5}),
    ("A2", "+ спектральные (PhaseSpectrumNoise)", {"phase_spectrum_noise": 0.5}),
    ("A3", "+ аддитивный шум (RandomJitter)", {"random_jitter": 0.6}),
    ("A4", "+ амплитудные (RandomInvert, ChannelScaling)", {"random_invert": 0.5, "channel_scaling": 0.2}),
    ("A5", "+ маскирование (TimeMasking, ChannelMasking)", {"time_masking": 0.2, "channel_masking": 0.2}),
    ("A6", "+ композитные (mosaic, mix-up)", {"mosaic": 0.5, "mix_up": 0.5}),
    ("A7", "+ синусоидальные помехи (SineNoise)", {"sine_noise": 0.5}),
    ("A8", "+ прямоугольные шумы (SquareNoises) = полный набор", {"square_noises": 0.1}),
]
INPUT_CONFIGS = [("IN_raw", "raw", "только «грязная» половина (15 каналов)"),
                 ("IN_filtered", "filtered", "только «чистая» половина (15 каналов)")]
METRIC_KEYS = ["mAP", "mAP@50", "Precision", "Recall"]


# ------------------------------------------------------------------ конфиги

def ablation_augmentations(group_id: str) -> dict:
    """Параметры аугментаций кумулятивного конфига `group_id` (A0…A8)."""
    ids = [g[0] for g in GROUPS]
    if group_id not in ids:
        raise KeyError(f"нет группы {group_id!r}, есть {ids}")
    aug = dict(AUG_OFF)
    for gid, _, delta in GROUPS[:ids.index(group_id) + 1]:
        aug.update(delta)
    return aug


def make_config(config_id: str, in_base: str = "A8") -> dict:
    """Полный конфиг A0…A8 или IN_raw / IN_filtered (на базе `in_base`)."""
    training = copy.deepcopy(BASE_TRAINING)
    inputs = {c[0]: c for c in INPUT_CONFIGS}
    if config_id in inputs:
        _, subset, text = inputs[config_id]
        training["augmentations"] = ablation_augmentations(in_base)
        training["input_subset"] = subset
        description = f"Абляция входа (E6): {text}; аугментации — {in_base}."
    else:
        training["augmentations"] = ablation_augmentations(config_id)
        description = f"Кумулятивная абляция: {dict((g[0], g[1]) for g in GROUPS)[config_id]}."
    return {"name": config_id, "description": description, "dataset": DATASET, "seeds": list(SEEDS),
            "training": training}


def write_configs(out_dir: Path = CONFIG_DIR, in_base: str = "A8") -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cid in [g[0] for g in GROUPS] + [c[0] for c in INPUT_CONFIGS]:
        path = out_dir / f"{cid}.yaml"
        header = "# Сгенерировано: python run_ec_ablation.py make-configs (таблица GROUPS). Правки — в run_ec_ablation.py.\n"
        path.write_text(header + yaml.safe_dump(make_config(cid, in_base), allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        paths.append(path)
    return paths


def check_training_keys(training: dict) -> None:
    """Неизвестные поля TrainingSettings / AugmentationSettings — ошибка (pydantic молча игнорирует их)."""
    unknown = set(training) - set(TrainingSettings.model_fields)
    unknown |= {f"augmentations.{k}" for k in set(training.get("augmentations", {})) - set(AugmentationSettings.model_fields)}
    forbidden = {"dataset", "results_path", "project", "seed"} & set(training)
    if unknown:
        raise ValueError(f"неизвестные поля конфига: {sorted(unknown)}")
    if forbidden:
        raise ValueError(f"поля {sorted(forbidden)} задаёт раннер, уберите их из training")


def load_config(config_id: str, config_dir: Path = CONFIG_DIR, epochs: Optional[int] = None) -> dict:
    path = config_dir / f"{config_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: нет конфига, выполните `python run_ec_ablation.py make-configs`")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if epochs is not None:
        config["training"]["epochs"] = epochs
    check_training_keys(config["training"])
    return config


# ------------------------------------------------------------------ обучение и оценка

def _metrics(r: dict, per_class, names: List[str]) -> dict:
    """Результат `Trainer._evaluate_loader` → JSON-совместимый словарь."""
    out = {k: float(r.get(k, 0.0)) for k in METRIC_KEYS}
    pre, rec, ap50, ap = (np.asarray(x, float) for x in per_class)
    if len(ap) == len(names):
        out["per_class"] = {n: {"AP": float(a), "AP@50": float(a50), "Precision": float(p), "Recall": float(q)}
                            for n, p, q, a50, a in zip(names, pre, rec, ap50, ap)}
    else:   # в сплите есть не все классы — compute_ap возвращает только присутствующие
        out["per_class_unnamed"] = {"AP": ap.tolist(), "AP@50": ap50.tolist()}
    return out


def save_predictions(path: Path, ids: List[str], per_sample) -> None:
    """Детекции [x1, x2, conf, cls] (отсчёты входа) и разметка [cls], [center, width] по каждому сигналу."""
    def ragged(arrays):   # массив объектов: np.asarray склеил бы массивы одинаковой формы в один
        out = np.empty(len(arrays), dtype=object)
        out[:] = arrays
        return out

    det = [np.asarray(d, np.float32).reshape(-1, 4) for d, _, _ in per_sample]
    cls = [np.asarray(c, np.float32).reshape(-1) for _, c, _ in per_sample]
    box = [np.asarray(b, np.float32).reshape(-1, 2) for _, _, b in per_sample]
    np.savez_compressed(path, ids=np.asarray(ids), det=ragged(det), cls=ragged(cls), box=ragged(box))


def load_predictions(path: Path) -> dict:
    """Обратное к `save_predictions`: {"ids", "det", "cls", "box"} (списки по сигналам)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: list(z[k]) for k in ("ids", "det", "cls", "box")}


def make_trainer(config: dict, run_dir: Path, seed: int, resume: bool):
    from yolo_1d.yolo_1d import Trainer
    settings = TrainingSettings(**config["training"], dataset=str(REPO / config["dataset"]),
                                results_path=str(run_dir), project="train", seed=seed, resume=resume)
    return Trainer(settings)


def _load_best(trainer) -> str:
    best = Path(trainer._best_path)
    if best.is_file():
        trainer.load_checkpoint(str(best))
        return best.name
    return "last (best.pt не создан)"


def run_one(config: dict, seed: int, runs_dir: Path = RUNS_DIR, rerun: bool = False) -> Path:
    """Обучение и оценка одного конфига с одним seed."""
    run_dir = runs_dir / f"{config['name']}_s{seed}"
    if (run_dir / "metrics.json").is_file() and not rerun:
        print(f"[skip] {run_dir.name}: уже есть metrics.json")
        return run_dir
    if rerun and run_dir.exists():
        shutil.rmtree(run_dir)
    resume = (run_dir / "train1" / "last.pt").is_file()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")

    start = time.time()
    trainer = make_trainer(config, run_dir, seed, resume)
    trainer.train()
    checkpoint = _load_best(trainer)
    names = config["training"]["names"]
    result = {}
    for split in ("val", "test"):
        trainer._init_dataloaders()
        loader = {"val": trainer._val_loader, "test": trainer._test_loader}[split]
        with torch.no_grad():
            r, per_class, per_sample = trainer._evaluate_loader(loader, desc=split)
        result[split] = _metrics(r, per_class, names)
        if split == "test":
            ids = [Path(f).stem for f in loader.dataset.filenames]
            save_predictions(run_dir / "predictions_test.npz", ids, per_sample)
    (run_dir / "metrics.json").write_text(json.dumps({"config": config["name"], "seed": seed, **result}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({"checkpoint": checkpoint, "resumed": resume,
                                                  "duration_s": round(time.time() - start, 1)}, indent=2),
                                      encoding="utf-8")
    torch.cuda.empty_cache()
    return run_dir


# ------------------------------------------------------------------ E5: test + SquareNoises

def noisy_test_transform(settings: AugmentationSettings, cache_dir: Optional[str], noise_seed: int = 0):
    """Валидационный конвейер с прямоугольной помехой на шаге 6 (p = 1).

    Помеха детерминирована: seed — от содержимого сигнала и `noise_seed` (не зависит от порядка и воркеров).
    Полный результат кэшируется отдельно от чистого test (`<cache>/noisy_test_s<noise_seed>/`).
    """
    from torchvision.transforms.v2 import Compose
    from yolo_1d.utils.transforms import EddyPreprocessing, PreprocessedCache, ReferenceCache, SquareNoises
    from yolo_1d.utils.transforms.transforms import Padding, Resize

    class SeededSquareNoises(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = SquareNoises()

        def forward(self, signals, bboxes, labels):
            key = ReferenceCache.key(signals.numpy()) + str(noise_seed)
            state = np.random.get_state()
            np.random.seed(int(key[:8], 16) ^ noise_seed)
            try:
                return self.noise(signals, bboxes, labels)
            finally:
                np.random.set_state(state)

    pre = EddyPreprocessing(cache_dir=cache_dir)
    ops = [pre.reference, SeededSquareNoises(), pre.ica, pre.destandardize, pre.fourier, pre.normalize]
    if cache_dir:
        ops = [PreprocessedCache(ops, str(Path(cache_dir) / f"noisy_test_s{noise_seed}"))]
    return Compose(ops + [Padding(settings.regular_length), Resize(settings.input_length)])


def run_noisy_test(config: dict, seed: int, runs_dir: Path = RUNS_DIR, noise_seed: int = 0) -> Path:
    from torch.utils.data import DataLoader
    from yolo_1d.utils.dataset import Dataset

    run_dir = runs_dir / f"{config['name']}_s{seed}"
    if not (run_dir / "metrics.json").is_file():
        raise FileNotFoundError(f"{run_dir}: прогон не завершён")
    trainer = make_trainer(config, run_dir, seed, resume=True)
    _load_best(trainer)
    cache = yaml.safe_load((REPO / config["dataset"]).read_text(encoding="utf-8")).get("preprocess_cache")
    aug = trainer._settings.augmentations
    dataset = Dataset(str(REPO / config["dataset"]), noisy_test_transform(aug, cache, noise_seed), "test")
    loader = DataLoader(dataset, trainer._settings.batch_size, shuffle=False, collate_fn=Dataset.collate_fn,
                        num_workers=trainer._settings.num_workers)
    with torch.no_grad():
        r, per_class, _ = trainer._evaluate_loader(loader, desc="noisy test")
    out = {"config": config["name"], "seed": seed, "noise_seed": noise_seed,
           "test_noisy": _metrics(r, per_class, config["training"]["names"])}
    (run_dir / "noisy_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.cuda.empty_cache()
    return run_dir


# ------------------------------------------------------------------ сводка

def collect(runs_dir: Path = RUNS_DIR) -> List[dict]:
    """Строки по прогонам: конфиг, seed, метрики val/test(/test_noisy), AP по классам на test."""
    rows = []
    for metrics_path in sorted(runs_dir.glob("*_s*/metrics.json")):
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"config": m["config"], "seed": m["seed"]}
        for split in ("val", "test"):
            row.update({f"{split}_{k}": m[split][k] for k in METRIC_KEYS})
        for name, v in m["test"].get("per_class", {}).items():
            row[f"test_AP_{name}"] = v["AP"]
            row[f"test_AP50_{name}"] = v["AP@50"]
        noisy = metrics_path.parent / "noisy_test.json"
        if noisy.is_file():
            n = json.loads(noisy.read_text(encoding="utf-8"))["test_noisy"]
            row.update({f"test_noisy_{k}": n[k] for k in METRIC_KEYS})
        rows.append(row)
    return rows


def _order(config: str) -> tuple:
    ids = [g[0] for g in GROUPS] + [c[0] for c in INPUT_CONFIGS]
    return (ids.index(config) if config in ids else len(ids), config)


def summarize(runs_dir: Path = RUNS_DIR, out_dir: Path = EXP_DIR) -> List[dict]:
    """summary.csv (по прогонам), summary_agg.csv (mean ± SD по seed), summary.md."""
    rows = sorted(collect(runs_dir), key=lambda r: (_order(r["config"]), r["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "seed"), k))
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    agg = []
    for config in sorted({r["config"] for r in rows}, key=_order):
        runs = [r for r in rows if r["config"] == config]
        a = {"config": config, "n_seeds": len(runs)}
        for k in fields[2:]:
            vals = [r[k] for r in runs if k in r]
            if vals:
                a[f"{k}_mean"], a[f"{k}_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg.append(a)
    agg_fields = sorted({k for a in agg for k in a}, key=lambda k: (k not in ("config", "n_seeds"), k))
    with open(out_dir / "summary_agg.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg)
    lines = ["# Абляция аугментаций ECRv1", "", "Test, mean ± SD по seed (mAP — mAP@0.5:0.95, макро по 6 классам).", "",
             "| конфиг | seeds | mAP | mAP@50 | P | R | mAP, test + шум |", "|---|---|---|---|---|---|---|"]
    fmt = lambda a, k: (f"{100 * a[k + '_mean']:.2f} ± {100 * a[k + '_sd']:.2f}" if k + "_mean" in a else "—")
    for a in agg:
        lines.append(f"| {a['config']} | {a['n_seeds']} | {fmt(a, 'test_mAP')} | {fmt(a, 'test_mAP@50')} | "
                     f"{fmt(a, 'test_Precision')} | {fmt(a, 'test_Recall')} | {fmt(a, 'test_noisy_mAP')} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    # при перенаправлении в файл Windows по умолчанию пишет в cp1251, а torchinfo печатает псевдографику
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make-configs", help="сгенерировать yaml-конфиги A0…A8, IN_raw, IN_filtered")
    p.add_argument("--in-base", default="A8", help="база аугментаций для IN_raw / IN_filtered (по умолчанию A8)")
    for name in ("train", "noisy-test"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="+", help="ID конфигов (A0 … A8, IN_raw, IN_filtered)")
        p.add_argument("--seeds", type=int, nargs="+", help="переопределить seeds конфига")
        p.add_argument("--epochs", type=int, help="переопределить training.epochs (смоук)")
        if name == "train":
            p.add_argument("--rerun", action="store_true", help="обучить заново, удалив прежний прогон")
        else:
            p.add_argument("--noise-seed", type=int, default=0, help="seed зашумлённой копии test")
    sub.add_parser("summary", help="пересобрать сводку по готовым прогонам")
    args = parser.parse_args(argv)

    if args.command == "make-configs":
        for path in write_configs(in_base=args.in_base):
            print(path.relative_to(REPO))
        return 0
    if args.command == "summary":
        summarize()
        print((EXP_DIR / "summary.md").read_text(encoding="utf-8"))
        return 0
    for cid in args.configs:
        config = load_config(cid, epochs=args.epochs)
        for seed in args.seeds if args.seeds is not None else config["seeds"]:
            print(f"\n===== {cid}, seed {seed} ({args.command}) =====")
            if args.command == "train":
                run_one(config, seed, rerun=args.rerun)
            else:
                run_noisy_test(config, seed, noise_seed=args.noise_seed)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
