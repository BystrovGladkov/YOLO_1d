"""E1: профиль обучающего конвейера A8 по шагам (один процесс, BLAS = 1 поток).

    python experiments/ec_ablation/e1_profile.py --n 40
"""
import argparse, sys, time, random
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_bench import log
from run_ec_ablation import load_config
from yolo_1d.utils.dataset import Dataset
from yolo_1d.utils.settings import AugmentationSettings
from yolo_1d.utils.transforms import build_transforms_for_dataset

p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=40); p.add_argument("--config", default="A8"); a = p.parse_args()
cfg = load_config(a.config); aug = AugmentationSettings(**cfg["training"]["augmentations"])
tr, _ = build_transforms_for_dataset(aug, str(REPO / cfg["dataset"]))
acc = defaultdict(float)
def wrap(op):
    name = type(getattr(op, "augmentation", op)).__name__
    f = op.forward
    def timed(*x):
        t = time.perf_counter(); r = f(*x); acc[name] += time.perf_counter() - t; return r
    op.forward = timed
for op in tr.transforms: wrap(op)
ds = Dataset(str(REPO / cfg["dataset"]), tr, "train", mosaic_prob=aug.mosaic, mix_up_prob=aug.mix_up)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
t0 = time.perf_counter()
for i in np.random.default_rng(0).choice(ds.n, a.n, replace=False): ds[int(i)]
total = time.perf_counter() - t0
steps = {k: round(v / a.n, 3) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
log({"kind": "profile", "config": a.config, "samples": a.n, "s_per_sample": round(total / a.n, 3),
     "transform_s_per_sample": round(sum(acc.values()) / a.n, 3), "per_step_s": steps})
