"""E4: парный bootstrap по сигналам test для кумулятивной абляции (R1 §3).

Для каждого прогона `runs/<ID>_s<seed>/predictions_test.npz` (детекции и разметка по сигналам) один раз
считаются TP-матрицы (`util.compute_metric`, IoU 0.5:0.95) — так же, как в `Trainer._evaluate_loader`.
Затем B раз выбираются 294 сигнала с возвращением; **одна и та же выборка** используется для всех конфигов,
на ней заново считается mAP (`util.compute_ap` по объединённым детекциям — mAP не среднее по сигналам).
Δ = mAP(шаг) − mAP(предыдущий шаг) и mAP(шаг) − mAP(A0) → 95 % перцентильный интервал и двусторонний
bootstrap p-value; поправка Холма на 8 последовательных шагов A0→A1 … A7→A8.

    python experiments/ec_ablation/e4_bootstrap.py --B 2000
Выход: experiments/ec_ablation/bootstrap.json, bootstrap.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from run_ec_ablation import EXP_DIR, GROUPS, NAMES, RUNS_DIR, all_config_ids, load_predictions  # noqa: E402
from yolo_1d.utils import util  # noqa: E402

IOU_V = torch.linspace(0.5, 0.95, 10)


def per_sample_metrics(run_dir: Path):
    """[(tp (n, 10) bool, conf (n,), pred_cls (n,), target_cls (m,))] по сигналам test."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    w = cfg["training"]["augmentations"]["input_length"]
    p = load_predictions(run_dir / "predictions_test.npz")
    out = []
    for det, cls, box in zip(p["det"], p["cls"], p["box"]):
        det = torch.as_tensor(np.asarray(det, np.float32).reshape(-1, 4))
        cls = torch.as_tensor(np.asarray(cls, np.float32).reshape(-1, 1))
        box = torch.as_tensor(np.asarray(box, np.float32).reshape(-1, 2))
        tp = torch.zeros(det.shape[0], IOU_V.numel(), dtype=torch.bool)
        if det.shape[0] and cls.shape[0]:
            target = torch.cat((cls, util.wh2xy(box) * w), dim=1)
            tp = util.compute_metric(det[:, :4], target, IOU_V)
        out.append((tp.numpy(), det[:, 2].numpy(), det[:, 3].numpy(), cls[:, 0].numpy()))
    return p["ids"], out


def map_of(samples, idx):
    """(mAP@0.5:0.95, mAP@0.5, AP по классам (6,) или NaN) на подвыборке сигналов idx."""
    tp = np.concatenate([samples[i][0] for i in idx])
    conf = np.concatenate([samples[i][1] for i in idx])
    pred = np.concatenate([samples[i][2] for i in idx])
    target = np.concatenate([samples[i][3] for i in idx])
    if not tp.any():
        return 0.0, 0.0, np.full(len(NAMES), np.nan)
    *_, ap50, ap = util.compute_ap(tp, conf, pred, target)
    per_class = np.full(len(NAMES), np.nan)
    per_class[np.unique(target).astype(int)] = ap
    return float(ap.mean()), float(ap50.mean()), per_class


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def interval(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(lo), float(hi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="seed обучения (прогоны <ID>_s<seed>)")
    ap.add_argument("--rng", type=int, default=12345)
    a = ap.parse_args()

    configs = [c for c in all_config_ids()
               if (RUNS_DIR / f"{c}_s{a.seed}" / "predictions_test.npz").is_file()]
    data, ids0 = {}, None
    for c in configs:
        ids, data[c] = per_sample_metrics(RUNS_DIR / f"{c}_s{a.seed}")
        ids0 = ids if ids0 is None else ids0
        assert list(ids) == list(ids0), f"{c}: другой порядок сигналов test"
    n = len(ids0)
    full = {c: map_of(data[c], np.arange(n)) for c in configs}
    # проверка: пересчёт совпадает с metrics.json прогона
    for c in configs:
        m = json.loads((RUNS_DIR / f"{c}_s{a.seed}" / "metrics.json").read_text(encoding="utf-8"))["test"]
        assert abs(full[c][0] - m["mAP"]) < 1e-6 and abs(full[c][1] - m["mAP@50"]) < 1e-6, (c, full[c][:2], m["mAP"])

    rng = np.random.default_rng(a.rng)
    boot = {c: np.empty(a.B) for c in configs}
    boot_cls = {c: np.empty((a.B, len(NAMES))) for c in configs}
    for b in range(a.B):
        idx = rng.integers(0, n, n)
        for c in configs:
            boot[c][b], _, boot_cls[c][b] = map_of(data[c], idx)

    steps = [c for c in [g[0] for g in GROUPS] if c in configs]
    res = {"B": a.B, "n_signals": n, "seed": a.seed, "configs": {}, "steps": [], "vs_A0": [], "inputs": []}
    for c in configs:
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        res["configs"][c] = {"mAP": full[c][0], "mAP@50": full[c][1], "ci": [float(lo), float(hi)],
                             "AP": dict(zip(NAMES, map(float, full[c][2])))}

    def compare(x, y):
        d = boot[y] - boot[x]
        lo, hi, p = interval(d)
        dc = boot_cls[y] - boot_cls[x]
        per_class = {}
        for k, name in enumerate(NAMES):
            v = dc[:, k][~np.isnan(dc[:, k])]
            if len(v):
                clo, chi, cp = interval(v)
                per_class[name] = {"delta": float(full[y][2][k] - full[x][2][k]), "ci": [clo, chi], "p": cp}
        return {"from": x, "to": y, "delta": full[y][0] - full[x][0], "ci": [lo, hi], "p": p, "per_class": per_class}

    res["steps"] = [compare(x, y) for x, y in zip(steps, steps[1:])]
    if res["steps"]:
        for s, padj in zip(res["steps"], holm(np.array([s["p"] for s in res["steps"]]))):
            s["p_holm"] = float(padj)
    res["vs_A0"] = [compare("A0", y) for y in steps[1:]]
    base = steps[-1] if steps else None
    res["inputs"] = [compare(base, c) for c in configs if c.startswith("IN_") and base]
    res["leave_out"] = [compare(base, c) for c in configs if c.startswith("A8_") and base]

    (EXP_DIR / "bootstrap.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    pct = lambda v: f"{100 * v:+.2f}"
    lines = [f"# Парный bootstrap по сигналам test (n = {n}, B = {a.B}, seed обучения {a.seed})", "",
             "mAP@0.5:0.95 (×100), 95 % перцентильный интервал. p — двусторонний bootstrap, p_Holm — на шаги.", "",
             "| конфиг | mAP | 95 % ДИ |", "|---|---|---|"]
    for c, v in res["configs"].items():
        lines.append(f"| {c} | {100 * v['mAP']:.2f} | {100 * v['ci'][0]:.2f} … {100 * v['ci'][1]:.2f} |")
    lines += ["", "| шаг | Δ mAP | 95 % ДИ | p | p_Holm | Δ от A0 | 95 % ДИ |", "|---|---|---|---|---|---|---|"]
    for s, v in zip(res["steps"], res["vs_A0"]):
        lines.append(f"| {s['from']}→{s['to']} | {pct(s['delta'])} | {pct(s['ci'][0])} … {pct(s['ci'][1])} | "
                     f"{s['p']:.3f} | {s['p_holm']:.3f} | {pct(v['delta'])} | {pct(v['ci'][0])} … {pct(v['ci'][1])} |")
    lines += ["", "Δ AP по классам на каждом шаге (×100; * — 95 % ДИ не содержит 0):", "",
              "| шаг | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    for s in res["steps"] + res["inputs"] + res["leave_out"]:
        cells = []
        for name in NAMES:
            v = s["per_class"].get(name)
            cells.append("—" if v is None else f"{pct(v['delta'])}{'*' if v['ci'][0] > 0 or v['ci'][1] < 0 else ''}")
        lines.append(f"| {s['from']}→{s['to']} | " + " | ".join(cells) + " |")
    if res["inputs"]:
        lines += ["", "| вход | Δ mAP к " + base + " | 95 % ДИ | p |", "|---|---|---|---|"]
        for s in res["inputs"]:
            lines.append(f"| {s['to']} | {pct(s['delta'])} | {pct(s['ci'][0])} … {pct(s['ci'][1])} | {s['p']:.3f} |")
    if res["leave_out"]:
        lines += ["", "| без группы | Δ mAP к " + base + " | 95 % ДИ | p |", "|---|---|---|---|"]
        for s in res["leave_out"]:
            lines.append(f"| {s['to']} | {pct(s['delta'])} | {pct(s['ci'][0])} … {pct(s['ci'][1])} | {s['p']:.3f} |")
    (EXP_DIR / "bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
