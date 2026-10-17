"""E1, открытый вопрос 1: ломает ли mosaic-склейка ICA (шаг 7)?

Обучающий конвейер A8 на N примерах с mosaic_prob = 1 и на N обычных (mosaic_prob = 0, mix-up выключен).
Считаются: предупреждения FastICA о несходимости (ConvergenceWarning, все, без схлопывания повторов),
прочие предупреждения, NaN/Inf, доля клипа/экстремальных значений на выходе.

    python experiments/ec_ablation/e1_ica_check.py --n 150
Результат — строка в experiments/ec_ablation/e1/bench.jsonl.
"""
import argparse
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_bench import log  # noqa: E402
from run_ec_ablation import load_config  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import build_transforms_for_dataset  # noqa: E402


def check(mosaic: bool, n: int, seed: int) -> dict:
    config = load_config("A8")
    aug = AugmentationSettings(**config["training"]["augmentations"])
    dataset = str(REPO / config["dataset"])
    train_t, _ = build_transforms_for_dataset(aug, dataset)
    ds = Dataset(dataset, train_t, "train", mosaic_prob=1.0 if mosaic else 0.0, mix_up_prob=0.0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).choice(ds.n, size=n, replace=False)
    kinds, nonfinite, absmax, n_warn_samples = Counter(), 0, [], 0
    for i in idx:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = ds[int(i)][0]
        names = [type(m.message).__name__ + ": " + str(m.message)[:60] for m in w]
        kinds.update(names)
        n_warn_samples += bool(w)
        nonfinite += int(not torch.isfinite(x).all())
        absmax.append(float(x.abs().max()))
    conv = sum(v for k, v in kinds.items() if k.startswith("ConvergenceWarning"))
    return {"kind": "ica_check", "mosaic": mosaic, "samples": n, "convergence_warnings": conv,
            "samples_with_warnings": n_warn_samples, "warning_kinds": dict(kinds.most_common(5)),
            "nonfinite_samples": nonfinite, "absmax_median": round(float(np.median(absmax)), 2),
            "absmax_p99": round(float(np.percentile(absmax, 99)), 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for mosaic in (False, True):
        log(check(mosaic, a.n, a.seed))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
