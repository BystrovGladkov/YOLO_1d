"""E2: сводка прогонов этапа — val (лучшая эпоха из step.csv) и test (metrics.json) по фолдам.

Использование: python experiments/scripts/e2_collect.py e2s1_A_nano e2s1_B_nano ... [--out file.md]
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def collect(name):
    rows = []
    for run in sorted((REPO / "experiments" / name).glob("fold*_seed*")):
        if not (run / "run.json").is_file():
            continue
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
        step = pd.read_csv(run / info["train_dir"] / "step.csv")
        best = step.loc[step["F1_mean6"].idxmax()]
        test = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["test"]
        pts = test["points"]
        rows.append({"exp": name, "run": run.name, "epochs": len(step), "best_epoch": int(best["epoch"]),
                     "val_F1": best["F1_mean6"], "val_sd": best["sd_mean6_ms"],
                     "train_loss_end": float((step.box_train + step.cls_train + step.dfl_train).iloc[-1]),
                     "val_loss_min": float((step.box_val + step.cls_val + step.dfl_val).min()),
                     "test_F1": test["summary"]["F1_mean6"], "test_sd": test["summary"]["sd_mean6_ms"],
                     "test_F1_P": (pts["P_on"]["F1"] + pts["P_off"]["F1"]) / 2,
                     "test_F1_QRS": (pts["QRS_on"]["F1"] + pts["QRS_off"]["F1"]) / 2,
                     "test_F1_T": (pts["T_on"]["F1"] + pts["T_off"]["F1"]) / 2,
                     "minutes": info["duration_s"] / 60})
    return rows


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    df = pd.DataFrame([r for n in names for r in collect(n)])
    mean = df.groupby("exp", sort=False).mean(numeric_only=True)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("\nСреднее по фолдам:")
    print(mean[["val_F1", "val_sd", "test_F1", "test_sd", "test_F1_P", "test_F1_QRS", "test_F1_T", "best_epoch", "minutes"]].round(4).to_string())
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.write_text(mean.round(4).to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    main()
