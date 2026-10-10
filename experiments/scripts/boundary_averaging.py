"""Гипотеза «усреднения границ» (R5 §1, E6): ставит ли модель границу волны ближе к медиане разметки этой волны
по всем отведениям записи, чем к разметке своего отведения?

Для каждого совпавшего onset/offset (детекция того же класса, центр ≤ 150 мс от центра волны эксперта) считаются
|pred − own| (ошибка относительно своего отведения) и |pred − median| (относительно медианы той же волны по
отведениям записи; волна в других отведениях — того же класса, центр ≤ 60 мс). Доля «ближе к медиане» > 0.5
и ⟨|pred − median|⟩ < ⟨|pred − own|⟩ — признак усреднения.
    python experiments/scripts/boundary_averaging.py experiments/baseline experiments/e6_bs8_lr01 --run fold0_seed0
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import apply_thresholds  # noqa: E402

NAMES = ["P", "QRS", "T"]


def analyse(run_dir: Path):
    th = {k: (v["conf"], v["val_F1"]) for k, v in json.loads((run_dir / "thresholds.json").read_text()).items()}
    samples = json.loads((run_dir / "predictions_test.json").read_text())
    by_rec = defaultdict(list)
    for s in samples:
        by_rec[s["id"].rsplit("_", 1)[0]].append(s)
    out = defaultdict(lambda: {"own": [], "med": [], "closer": []})
    for leads in by_rec.values():
        refs = [np.asarray(s["ref"], float).reshape(-1, 3) for s in leads]
        for s, ref in zip(leads, refs):
            pred = apply_thresholds(np.asarray(s["pred"], float).reshape(-1, 4), th, NAMES)
            for c, name in enumerate(NAMES):
                rc, pc = ref[ref[:, 0] == c], pred[pred[:, 0] == c]
                for w in rc:
                    centre = (w[1] + w[2]) / 2
                    if not len(pc):
                        continue
                    j = np.argmin(np.abs((pc[:, 1] + pc[:, 2]) / 2 - centre))
                    if abs((pc[j, 1] + pc[j, 2]) / 2 - centre) > 75:
                        continue
                    same = [r[(r[:, 0] == c) & (np.abs((r[:, 1] + r[:, 2]) / 2 - centre) <= 30)] for r in refs]
                    same = np.concatenate([x for x in same if len(x)])
                    for col in (1, 2):
                        med = np.median(same[:, col])
                        if abs(med - w[col]) < 2:  # своя граница совпадает с медианой — не информативно
                            continue
                        e_own, e_med = abs(pc[j, col] - w[col]), abs(pc[j, col] - med)
                        key = f"{name}_{'on' if col == 1 else 'off'}"
                        out[key]["own"].append(e_own * 2)
                        out[key]["med"].append(e_med * 2)
                        out[key]["closer"].append(e_med < e_own)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+", type=Path)
    p.add_argument("--run", default="fold0_seed0")
    a = p.parse_args()
    print("| Эксперимент | Точка | n | ⟨|pred − своё|⟩, мс | ⟨|pred − медиана|⟩, мс | доля ближе к медиане |\n|---|---|---|---|---|---|")
    for e in a.experiments:
        for k, v in sorted(analyse(e / a.run).items()):
            print(f"| {e.name} | {k} | {len(v['own'])} | {np.mean(v['own']):.1f} | {np.mean(v['med']):.1f} | "
                  f"{100 * np.mean(v['closer']):.0f} % |")


if __name__ == "__main__":
    main()
