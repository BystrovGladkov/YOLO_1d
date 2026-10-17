"""E1: замер скорости обучающего загрузчика ECRv1 (предобработка на лету) и шага модели на GPU.

    python experiments/ec_ablation/e1_bench.py loader --config A8 --workers 4 8 12 --batches 20
    python experiments/ec_ablation/e1_bench.py gpu --batches 20

Результаты дописываются в experiments/ec_ablation/e1/bench.jsonl (одна строка JSON на замер).
Предупреждения FastICA (ConvergenceWarning) печатаются воркерами в stderr — запускать с перенаправлением
stderr в лог и считать их там; NaN/Inf в батчах считаются здесь.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from run_ec_ablation import EXP_DIR, load_config  # noqa: E402
from run_experiment import environment  # noqa: E402
from yolo_1d.utils import util  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import build_transforms_for_dataset  # noqa: E402
from yolo_1d.yolo_1d import _seed_worker  # noqa: E402

OUT = EXP_DIR / "e1" / "bench.jsonl"


def log(record: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


def bench_loader(config_id: str, workers: int, batches: int, batch_size: int, tag: str) -> None:
    config = load_config(config_id)
    aug = AugmentationSettings(**config["training"]["augmentations"])
    dataset = str(REPO / config["dataset"])
    train_t, _ = build_transforms_for_dataset(aug, dataset)
    ds = Dataset(dataset, train_t, "train", mosaic_prob=aug.mosaic, mix_up_prob=aug.mix_up,
                 mix_up_alpha=aug.mix_up_alpha)
    sampler = WeightedRandomSampler(util.get_data_weights(ds), num_samples=ds.n, replacement=True)
    kw = dict(num_workers=workers, collate_fn=Dataset.collate_fn, pin_memory=True)
    if workers > 0:
        kw.update(persistent_workers=False, worker_init_fn=_seed_worker)
    loader = DataLoader(ds, batch_size, sampler=sampler, **kw)
    n_total = len(loader) if batches <= 0 else min(batches, len(loader))
    t0 = time.time()
    it = iter(loader)
    times, bad = [], 0
    for i in range(n_total):
        x, _ = next(it)
        times.append(time.time())
        bad += int((~torch.isfinite(x)).any(dim=(1, 2)).sum())
    del it
    # первый батч включает запуск воркеров; скорость — по остальным
    startup = times[0] - t0
    steady = (times[-1] - times[0]) / max(len(times) - 1, 1)
    log({"kind": "loader", "tag": tag, "config": config_id, "workers": workers, "batch_size": batch_size,
         "batches": n_total, "startup_s": round(startup, 1), "s_per_batch": round(steady, 2),
         "s_per_sample": round(steady / batch_size, 3), "epoch_min_est": round(steady * len(loader) / 60, 1),
         "nonfinite_samples": bad, "mosaic": aug.mosaic, "mix_up": aug.mix_up})


def bench_gpu(batches: int, batch_size: int, channels: int, model_size: str) -> None:
    """Шаг обучения (forward + backward + optimizer) на синтетическом батче — нижняя граница времени эпохи."""
    config = load_config("A0")
    from run_ec_ablation import make_trainer
    run_dir = EXP_DIR / "e1" / "gpu_bench"
    trainer = make_trainer({**config, "training": {**config["training"], "model_size": model_size,
                                                   "batch_size": batch_size}}, run_dir, 0, False)
    trainer._init_dataloaders()
    trainer._init_training()
    model, criterion, opt, scaler = trainer._model, trainer._criterion, trainer._optimizer, trainer._amp_scale
    x, targets = next(iter(trainer._train_loader))
    x = x.cuda().float()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(batches):
        torch.cuda.synchronize()
        t = time.time()
        with torch.amp.autocast("cuda"):
            out = model(x)
            lb, lc, ld = criterion(out, targets)
        scaler.scale((lb + lc + ld) * batch_size).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    steady = sum(times[3:]) / max(len(times) - 3, 1)
    steps = len(trainer._train_loader)
    log({"kind": "gpu", "model_size": model_size, "batch_size": batch_size, "channels": x.shape[1],
         "s_per_step": round(steady, 3), "steps_per_epoch": steps, "epoch_min_est": round(steady * steps / 60, 2),
         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "params_m": round(
             sum(p.numel() for p in model.parameters()) / 1e6, 2), "env": environment()["gpu"]})


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("loader")
    pl.add_argument("--config", default="A8")
    pl.add_argument("--workers", type=int, nargs="+", default=[4])
    pl.add_argument("--batches", type=int, default=20, help="0 — вся эпоха")
    pl.add_argument("--batch-size", type=int, default=50)
    pl.add_argument("--tag", default="")
    pg = sub.add_parser("gpu")
    pg.add_argument("--batches", type=int, default=20)
    pg.add_argument("--batch-size", type=int, default=50)
    pg.add_argument("--model-size", default="large")
    a = p.parse_args()
    if a.cmd == "loader":
        for w in a.workers:
            bench_loader(a.config, w, a.batches, a.batch_size, a.tag)
    else:
        bench_gpu(a.batches, a.batch_size, 30, a.model_size)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    main()
