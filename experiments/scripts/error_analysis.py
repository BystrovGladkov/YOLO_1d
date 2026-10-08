"""Анализ ошибок итоговой модели (E4): метрики по записям, диагнозам и отведениям, худшие записи,
положение ложных P.

Источник — сохранённые предсказания test каждого прогона (`predictions_test.json`, пороги `thresholds.json`),
т. е. каждая запись LUDB попадает в test ровно одного фолда; сиды объединяются (пулинг).
    python experiments/scripts/error_analysis.py experiments/final --out docs/results/e4
Вывод: <out>/by_category.md, by_lead.md, worst_records.md, false_p.md и records.csv.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import DelineationMetrics, apply_thresholds, match_points  # noqa: E402
from yolo_1d.utils.ludb import load_metadata, rhythm_group  # noqa: E402

FS, TOL = 500, 75  # Гц; допуск 150 мс в отсчётах
NAMES = ["P", "QRS", "T"]
LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]


def categories(meta: dict) -> list:
    """Диагностические категории записи LUDB (запись может входить в несколько)."""
    out = {"sinus": "Синусовый ритм (без ЭКС)", "af_flutter": "ФП / трепетание", "paced": "ЭКС"}
    cats = [out[rhythm_group(meta)]]
    cond = " ".join(meta.get("Conduction abnormalities", [])).lower()
    extra = " ".join(meta.get("Extrasystolies", [])).lower()
    if "iii degree" in cond:
        cats.append("AV-блокада III ст.")
    if "i degree" in cond.replace("iii degree", ""):
        cats.append("AV-блокада I ст.")
    if "complete" in cond.replace("incomplete", ""):
        cats.append("Полная блокада ножки (БЛНПГ/БПНПГ)")
    if "incomplete" in cond or "hemiblock" in cond or "intravintricular" in cond:
        cats.append("Неполная блокада / гемиблок / в/ж задержка")
    if "ventricular extrasystole" in extra:
        cats.append("Желудочковая экстрасистолия")
    if "atrial extrasystole" in extra:
        cats.append("Предсердная экстрасистолия")
    if meta.get("Ischemia"):
        cats.append("Ишемия / инфаркт / рубец")
    if meta.get("Hypertrophies"):
        cats.append("Гипертрофия / перегрузка")
    pathology = any(meta.get(k) for k in ("Conduction abnormalities", "Extrasystolies", "Hypertrophies",
                                          "Cardiac pacing", "Ischemia", "Non-specific repolarization abnormalities"))
    if rhythm_group(meta) == "sinus" and not pathology:
        cats.append("Без патологий (синус, без находок)")
    return cats


def load_runs(exp_dir: Path):
    """[(fold, seed, [(id, pred_thresholded, ref)])] для всех прогонов эксперимента."""
    runs = []
    for run_dir in sorted(exp_dir.glob("fold*_seed*")):
        if not (run_dir / "metrics.json").is_file():
            continue
        th = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        th = {k: (v["conf"], v["val_F1"]) for k, v in th.items()}
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        samples = []
        for s in json.loads((run_dir / "predictions_test.json").read_text(encoding="utf-8")):
            pred = apply_thresholds(np.asarray(s["pred"], float).reshape(-1, 4), th, NAMES)
            samples.append((s["id"], pred, np.asarray(s["ref"], float).reshape(-1, 3)))
        runs.append((info["fold"], info["seed"], samples))
    return runs


def row(name, res, extra=""):
    s, p = res["summary"], res["points"]
    f1 = lambda c: np.nanmean([p[f"{c}_on"]["F1"], p[f"{c}_off"]["F1"]]) * 100  # noqa: E731
    sd = lambda c: np.nanmean([p[f"{c}_on"]["sd_ms"], p[f"{c}_off"]["sd_ms"]])  # noqa: E731
    fp_p = p["P_on"].get("FP_per_lead", float("nan"))
    return (f"| {name}{extra} | {s['n_leads']} | {s['F1_mean6'] * 100:.2f} | {s['sd_mean6_ms']:.1f} | "
            f"{f1('P'):.2f} / {f1('QRS'):.2f} / {f1('T'):.2f} | {sd('P'):.1f} / {sd('QRS'):.1f} / {sd('T'):.1f} | "
            f"{fp_p:.2f} |")


HEADER = ("| Группа | отведений | ⟨F1⟩₆, % | ⟨σ⟩₆, мс | F1 P / QRS / T, % | σ P / QRS / T, мс | ложных P на отв. |\n"
          "|---|---|---|---|---|---|---|")


def false_p_positions(samples):
    """Ложные P (центр не ближе 150 мс к P эксперта, внутри размеченной зоны): расстояние до ближайшего
    последующего QRS эксперта (мс) и попадание внутрь T эксперта."""
    dist, in_t, count = [], 0, 0
    for _, pred, ref in samples:
        if not len(ref):
            continue
        lo, hi = ref[:, 1].min(), ref[:, 2].max()
        pp, rp = pred[pred[:, 0] == 0], ref[ref[:, 0] == 0]
        cp = (pp[:, 1] + pp[:, 2]) / 2
        pairs, _ = match_points(cp, (rp[:, 1] + rp[:, 2]) / 2, TOL)
        matched = set(int(i) for i in pairs[:, 0])
        qrs_on = np.sort(ref[ref[:, 0] == 1, 1])
        t = ref[ref[:, 0] == 2]
        for i, c in enumerate(cp):
            if i in matched or c < lo or c > hi:
                continue
            count += 1
            nxt = qrs_on[qrs_on > pp[i, 2]]
            if len(nxt):
                dist.append((nxt[0] - pp[i, 2]) * 1000 / FS)
            in_t += bool(((c >= t[:, 1]) & (c <= t[:, 2])).any())
    return np.array(dist), in_t, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(REPO / "data" / "raw" / "ludb")
    runs = load_runs(args.experiment)
    seeds = sorted({s for _, s, _ in runs})
    print(f"прогонов: {len(runs)}, сиды: {seeds}")

    new = lambda: DelineationMetrics(class_names=NAMES, exclude_boundary_points=True)  # noqa: E731
    by_cat, by_lead, by_rec = defaultdict(new), defaultdict(new), defaultdict(new)
    total = new()
    all_samples = defaultdict(list)
    for _, _, samples in runs:
        for sid, pred, ref in samples:
            rec, lead = sid.split("_")
            total.add(pred, ref)
            by_lead[lead].add(pred, ref)
            by_rec[rec].add(pred, ref)
            for c in categories(meta[rec]):
                by_cat[c].add(pred, ref)
                all_samples[c].append((sid, pred, ref))
            all_samples["all"].append((sid, pred, ref))

    note = f"Пулинг test всех фолдов, сиды {seeds} (каждое отведение учтено {len(seeds)} раз). Граничные точки исключены."
    cat_order = sorted(by_cat, key=lambda c: -by_cat[c].compute()["summary"]["F1_mean6"])
    lines = ["# Метрики по диагностическим категориям", "", note,
             "Запись может входить в несколько категорий; F1 P при ФП/трепетании не определён (P не размечены).", "",
             HEADER, row("**Все**", total.compute())]
    lines += [row(c, by_cat[c].compute(), f" ({len({s.split('_')[0] for s, _, _ in all_samples[c]})} зап.)")
              for c in cat_order]
    (args.out / "by_category.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    lines = ["# Метрики по отведениям", "", note, "", HEADER] + [row(l, by_lead[l].compute()) for l in LEADS]
    (args.out / "by_lead.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    recs = []
    for rec, m in by_rec.items():
        r = m.compute()
        p = r["points"]
        fp = sum(p[f]["FP"] for f in p) / 2
        fn = sum(p[f]["FN"] for f in p) / 2
        recs.append({"record": rec, "F1_mean6": r["summary"]["F1_mean6"], "sd_mean6_ms": r["summary"]["sd_mean6_ms"],
                     "FP": fp, "FN": fn, "rhythm": ", ".join(meta[rec]["Rhythms"]),
                     "categories": "; ".join(categories(meta[rec])[1:])})
    recs.sort(key=lambda x: x["F1_mean6"])
    with open(args.out / "records.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    lines = ["# Худшие записи по ⟨F1⟩₆", "", note, "FP/FN — среднее по 6 точкам, сумма по 12 отведениям и сидам.", "",
             "| Запись | ⟨F1⟩₆, % | ⟨σ⟩₆, мс | FP | FN | Ритм | Находки |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['record']} | {r['F1_mean6'] * 100:.1f} | {r['sd_mean6_ms']:.1f} | {r['FP']:.0f} | {r['FN']:.0f} | "
              f"{r['rhythm']} | {r['categories']} |" for r in recs[:15]]
    worst_sd = sorted(recs, key=lambda x: -x["sd_mean6_ms"])[:10]
    lines += ["", "## Худшие записи по ⟨σ⟩₆", "", "| Запись | ⟨σ⟩₆, мс | ⟨F1⟩₆, % | Ритм | Находки |", "|---|---|---|---|---|"]
    lines += [f"| {r['record']} | {r['sd_mean6_ms']:.1f} | {r['F1_mean6'] * 100:.1f} | {r['rhythm']} | {r['categories']} |"
              for r in worst_sd]
    (args.out / "worst_records.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    lines = ["# Ложные P: где они стоят", "", note,
             "Ложная P — центр дальше 150 мс от любой P эксперта, внутри размеченной зоны. "
             "«До QRS» — от конца ложной P до начала следующего QRS эксперта.", "",
             "| Группа | ложных P | на отведение | медиана до QRS, мс | доля 0–250 мс до QRS | внутри T эксперта |",
             "|---|---|---|---|---|---|"]
    for c in ["all", "ФП / трепетание", "ЭКС", "AV-блокада III ст.", "Синусовый ритм (без ЭКС)"]:
        d, in_t, total_fp = false_p_positions(all_samples[c])
        n = len(all_samples[c])
        lines.append(f"| {c} | {total_fp} | {total_fp / max(n, 1):.2f} | {np.median(d) if len(d) else float('nan'):.0f} | "
                     f"{np.mean((d >= 0) & (d <= 250)) * 100 if len(d) else float('nan'):.0f} % | {in_t} |")
    (args.out / "false_p.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
