"""Абляция ЭКГ-аугментаций (A2): кандидаты, генерация конфигов, сводная таблица и решения.

Протокол — experiments/ablation/PROTOCOL.md. Команды:

    # конфиги (пишутся в experiments/configs/abl_<...>.yaml на основе experiments/configs/baseline.yaml)
    python experiments/ablation/ablation.py configs full                 # все кандидаты (Full)
    python experiments/ablation/ablation.py configs groups               # Full без одной группы (уровень 1)
    python experiments/ablation/ablation.py configs items --groups N T   # Full без одной аугментации (уровень 2)
    python experiments/ablation/ablation.py configs add --ids N1 T2      # baseline + одна аугментация (add-one-in)
    python experiments/ablation/ablation.py configs final --drop N6 X1   # Full без отброшенных (подтверждение)

    # сводка: experiments/ablation/results.csv (по прогонам) и summary.md (сравнение и решения)
    python experiments/ablation/ablation.py summarize --ref abl_full [--exp abl_* ...]

Прогон конфигов — run_experiment.py (см. PROTOCOL.md §4).
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "experiments" / "configs"
BASE_CONFIG = CONFIGS / "baseline.yaml"
OUT = REPO / "experiments" / "ablation"

# Кандидаты (R3 §3, состав согласован в A1): ID → (группа, поля augmentations.ecg).
# Параметры — значения по умолчанию ECGAugmentationSettings; здесь задаются вероятности из R3.
CANDIDATES = {
    "N1": ("N", {"wander_p": 0.5}),
    "N2": ("N", {"powerline_p": 0.3}),
    "N3": ("N", {"emg_p": 0.3}),
    "N4": ("N", {"white_p": 0.3}),
    "N5": ("N", {"nstdb_p": 0.3}),
    "N6": ("N", {"step_p": 0.2}),
    "N7": ("N", {"filter_p": 0.3}),
    "T1": ("T", {"time_scale_p": 0.5}),
    "T2": ("T", {"hrv_warp_p": 0.5}),
    "T3": ("T", {"window_warp_p": 0.3}),
    "A1": ("A", {"amp_scale_p": 0.5}),
    "A2": ("A", {"wave_scale_p": 0.3}),
    "A3": ("A", {"invert_p": 0.3}),
    "M1": ("M", {"time_mask_p": 0.2}),
    "X1": ("X", {"mosaic_p": 0.2}),
    "X2": ("X", {"splice_p": 0.3}),
}
GROUPS = {g: [k for k, (grp, _) in CANDIDATES.items() if grp == g] for g in ("N", "T", "A", "M", "X")}
GROUP_NAMES = {"N": "шумы", "T": "время", "A": "амплитуда", "M": "маскирование", "X": "склейка записей"}

# пороги отбраковки (PROTOCOL.md §3): max(минимум из R2 §3.3, SD baseline по прогонам E2)
MIN_DELTA_F1, MIN_DELTA_SD = 0.002, 0.5


# --------------------------------------------------------------------------- конфиги
def ecg_settings(ids) -> dict:
    out = {}
    for i in ids:
        out.update(CANDIDATES[i][1])
    return out


def make_config(name: str, ids, description: str, folds, seeds) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["name"] = name
    config["description"] = description
    config["folds"], config["seeds"] = list(folds), list(seeds)
    # аугментации считаются на CPU: параллельные воркеры загрузки (только скорость; E3, решение пользователя)
    config["training"]["num_workers"] = 4
    aug = config["training"]["augmentations"]
    aug["pipeline"] = "ecg"
    aug["ecg"] = ecg_settings(ids)
    config["ablation"] = {"augmentations": sorted(ids)}  # для сводки; run_experiment это поле не использует
    path = CONFIGS / f"{name}.yaml"
    header = f"# Абляция аугментаций (experiments/ablation/PROTOCOL.md): {description}\n"
    path.write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def cmd_configs(args) -> None:
    all_ids = list(CANDIDATES)
    plan = []
    if args.level == "full":
        plan.append(("abl_full", all_ids, "Full — все кандидаты"))
    elif args.level == "groups":
        for g, ids in GROUPS.items():
            plan.append((f"abl_full_minus_{g}", [i for i in all_ids if i not in ids],
                         f"Full без группы {g} ({GROUP_NAMES[g]}: {', '.join(ids)})"))
    elif args.level == "items":
        for g in args.groups:
            for i in GROUPS[g]:
                plan.append((f"abl_full_minus_{i}", [k for k in all_ids if k != i], f"Full без {i}"))
    elif args.level == "add":
        for i in args.ids:
            plan.append((f"abl_base_plus_{i}", [i], f"baseline + {i}"))
    elif args.level == "final":
        keep = [i for i in all_ids if i not in set(args.drop)]
        plan.append(("abl_final", keep, f"Full без {', '.join(args.drop) or '—'} (итоговый набор)"))
    for name, ids, desc in plan:
        print(make_config(name, ids, desc, args.folds, args.seeds).relative_to(REPO))


# --------------------------------------------------------------------------- сводка
def run_rows(exp: str) -> list:
    """Строки results.csv: метрики test каждого прогона (фолд, сид) эксперимента."""
    exp_dir = REPO / "experiments" / exp
    config = yaml.safe_load((CONFIGS / f"{exp}.yaml").read_text(encoding="utf-8")) if (CONFIGS / f"{exp}.yaml").is_file() else {}
    augs = " ".join(config.get("ablation", {}).get("augmentations", [])) if config else ""
    rows = []
    for run in sorted(exp_dir.glob("fold*_seed*")):
        metrics_file = run / "metrics.json"
        if not metrics_file.is_file():
            continue
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        pts, s = m["test"]["points"], m["test"]["summary"]
        fold, seed = run.name.replace("fold", "").split("_seed")
        row = {"experiment": exp, "augmentations": augs, "fold": int(fold), "seed": int(seed),
               "F1_mean6": s["F1_mean6"], "sd_mean6_ms": s["sd_mean6_ms"], "mae_mean6_ms": s["mae_mean6_ms"]}
        for c in ("P", "QRS", "T"):
            row[f"F1_{c}"] = (pts[f"{c}_on"]["F1"] + pts[f"{c}_off"]["F1"]) / 2
            row[f"sd_{c}_ms"] = (pts[f"{c}_on"]["sd_ms"] + pts[f"{c}_off"]["sd_ms"]) / 2
        af = m.get("test_by_group", {}).get("af_flutter", {}).get("points", {}).get("P_on", {})
        row["FP_P_per_lead_af"] = af.get("FP_per_lead", np.nan)  # ложные P при ФП (есть в прогонах после M5)
        rows.append(row)
    return rows


def baseline_deltas() -> tuple:
    """Пороги δ_F1, δ_σ: max(минимум R2, SD baseline по прогонам E2)."""
    summary = json.loads((REPO / "experiments" / "baseline" / "summary.json").read_text(encoding="utf-8"))
    sd_f1 = float(np.std([r["F1_mean6"] for r in summary["runs"]]))
    sd_sd = float(np.std([r["sd_mean6_ms"] for r in summary["runs"]]))
    return max(MIN_DELTA_F1, sd_f1), max(MIN_DELTA_SD, sd_sd)


def decide(d_f1: float, d_sd: float, delta_f1: float, delta_sd: float, kind: str) -> str:
    """Решение по парной разнице (кандидат − эталон).

    kind="minus" (эталон Full, кандидат Full без X): X отбраковывается, если без неё
        F1 выше на ≥ δ_F1, или F1 не ниже (Δ ≥ −δ_F1/2) и σ меньше на ≥ δ_σ.
    kind="plus" (эталон baseline, кандидат baseline + X): X вредна, если с ней F1 ниже на ≥ δ_F1
        или σ больше на ≥ δ_σ при F1 не выше; полезна, если F1 выше на ≥ δ_F1 или σ ниже на ≥ δ_σ.
    Иначе — «нейтральна» (разница в пределах шума): X сохраняется.
    """
    if kind == "minus":
        if d_f1 >= delta_f1 or (d_f1 >= -delta_f1 / 2 and d_sd <= -delta_sd):
            return "отбросить"
        if d_f1 <= -delta_f1 or (d_f1 <= delta_f1 / 2 and d_sd >= delta_sd):
            return "оставить (полезна)"
        return "оставить (нейтральна)"
    if d_f1 <= -delta_f1 or (d_f1 <= 0 and d_sd >= delta_sd):
        return "вредна"
    if d_f1 >= delta_f1 or (d_f1 >= 0 and d_sd <= -delta_sd):
        return "полезна"
    return "нейтральна"


def paired(rows_ref: list, rows: list) -> tuple:
    """Парные разности по общим (фолд, сид): (n, ΔF1 mean, ΔF1 SD, Δσ mean, Δσ SD)."""
    ref = {(r["fold"], r["seed"]): r for r in rows_ref}
    pairs = [(r, ref[(r["fold"], r["seed"])]) for r in rows if (r["fold"], r["seed"]) in ref]
    if not pairs:
        return 0, np.nan, np.nan, np.nan, np.nan
    d_f1 = np.array([a["F1_mean6"] - b["F1_mean6"] for a, b in pairs])
    d_sd = np.array([a["sd_mean6_ms"] - b["sd_mean6_ms"] for a, b in pairs])
    return len(pairs), d_f1.mean(), d_f1.std(), d_sd.mean(), d_sd.std()


def cmd_summarize(args) -> None:
    exps = args.exp or sorted(p.name for p in (REPO / "experiments").glob("abl_*") if p.is_dir())
    names = ["baseline"] + [e for e in exps if e != "baseline"]
    rows = {e: run_rows(e) for e in names}
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(next(iter(r for rs in rows.values() for r in rs)).keys())
    with open(OUT / "results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rs in rows.values():
            writer.writerows(rs)

    delta_f1, delta_sd = baseline_deltas()
    lines = ["# Абляция аугментаций — сводка", "",
             f"Сгенерировано `experiments/ablation/ablation.py summarize --ref {args.ref}`. Протокол — [PROTOCOL.md](PROTOCOL.md). "
             f"Метрики — test каждого прогона, разности — **парные** по (фолд, сид). "
             f"Пороги: δ_F1 = {100 * delta_f1:.2f} п.п., δ_σ = {delta_sd:.2f} мс.", "",
             "| Эксперимент | прогонов | F1₆, % | σ₆, мс | F1 P / QRS / T, % | ложных P при ФП, на отв. | "
             f"ΔF1 к {args.ref}, п.п. | Δσ, мс | Решение |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in names:
        rs = rows[e]
        if not rs:
            continue
        def mean(key):
            values = np.asarray([r[key] for r in rs], float)
            return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")
        kind = "minus" if e.startswith("abl_full_minus_") else "plus" if e.startswith("abl_base_plus_") else None
        if e == args.ref and e != "baseline" or e == "abl_final":
            kind = "plus"  # шаг 0 / 3 протокола: эталон (Full) и итоговый набор сравниваются с baseline
        ref_name = "baseline" if kind == "plus" else args.ref
        n, df1, sdf1, dsd, sdsd = paired(rows.get(ref_name, []), rs) if e != ref_name else (0, np.nan, 0, np.nan, 0)
        decision = decide(df1, dsd, delta_f1, delta_sd, kind) if kind and n else ""
        delta = f"{100 * df1:+.2f} ± {100 * sdf1:.2f}" if n else "—"
        dsig = f"{dsd:+.2f} ± {sdsd:.2f}" if n else "—"
        if kind == "plus":
            delta += " (к baseline)"
        lines.append(f"| {e} | {len(rs)} | {100 * mean('F1_mean6'):.2f} | {mean('sd_mean6_ms'):.2f} | "
                     f"{100 * mean('F1_P'):.2f} / {100 * mean('F1_QRS'):.2f} / {100 * mean('F1_T'):.2f} | "
                     f"{mean('FP_P_per_lead_af'):.2f} | {delta} | {dsig} | {decision} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs")
    c.add_argument("level", choices=["full", "groups", "items", "add", "final"])
    c.add_argument("--groups", nargs="+", default=[], choices=list(GROUPS))
    c.add_argument("--ids", nargs="+", default=[], choices=list(CANDIDATES))
    c.add_argument("--drop", nargs="*", default=[], choices=list(CANDIDATES))
    c.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    c.add_argument("--seeds", type=int, nargs="+", default=[0])
    s = sub.add_parser("summarize")
    s.add_argument("--ref", default="abl_full")
    s.add_argument("--exp", nargs="+")
    args = parser.parse_args()
    {"configs": cmd_configs, "summarize": cmd_summarize}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
