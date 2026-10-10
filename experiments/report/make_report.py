"""P1: таблицы и графики для статьи → experiments/report/{tables,figures}/.

Все числа пересчитываются из логов экспериментов (predictions_test.json + thresholds.json каждого прогона,
step.csv, experiments/ablation/results.csv); литературные значения — из docs/research/R2_metrics_benchmarks.md §2.1.

Таблицы (Markdown + LaTeX/booktabs):
    T1_final_points        — итоговая модель (режим S, E4) и режим MC (E7): Se / PPV / F1 / m ± σ по 6 точкам
    T2_benchmarks          — сравнение с опубликованными методами на LUDB (F1 и m ± σ по точкам, ⟨F1⟩₆)
    T3_ablation            — абляция аугментаций (E3): baseline, Full, Full − группа; парные разности
    T4_modes               — режимы S и MC без и со всеми аугментациями (E2, E4, E7); парные разности MC − S
Графики (PNG, 200 dpi):
    F1_error_distributions — распределение ошибки (pred − ref) по 6 точкам: S (E4) и MC (E7)
    F2_learning_curves     — val ⟨F1⟩₆, val ⟨σ⟩₆ и лосс по шагам оптимизатора: S (E4) и MC (E7), медиана по прогонам
    F3_ablation            — парные разности ⟨F1⟩₆ и ⟨σ⟩₆ при удалении группы аугментаций (E3)
    F4_examples            — примеры делинеации: эксперт, S, MC на трёх записях test

    python experiments/report/make_report.py [--skip-examples]
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.delineation import DelineationMetrics, apply_thresholds  # noqa: E402

OUT = REPO / "experiments" / "report"
NAMES = ["P", "QRS", "T"]
POINTS = [f"{c}_{e}" for c in NAMES for e in ("on", "off")]
POINT_LABEL = {"P_on": "P on", "P_off": "P off", "QRS_on": "QRS on", "QRS_off": "QRS off", "T_on": "T on", "T_off": "T off"}
EXPERIMENTS = {"S": "final", "MC": "mc_full"}
LABEL = {"S": "MyNet-1d, S (1 отв.)", "MC": "MyNet-1d, MC (12 отв.)"}
# эталонная палитра (dataviz, references/palette.md, проверена validate_palette.js: 3 слота, все пары)
COLOR = {"S": "#2a78d6", "MC": "#eb6834", "P": "#1baf7a", "QRS": "#eb6834", "T": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# R2 §2.1: (метод, протокол, режим, {точка: (F1 %, m, σ)}, ⟨F1⟩) — F1* досчитаны в R2 по Se/PPV
LITERATURE = [
    ("Sereda 2019", "134 → 66 пациентов LUDB", "1 отв.",
     {"P_on": (88.49, 2.7, 21.9), "P_off": (88.53, -7.4, 28.6), "QRS_on": (98.84, 2.6, 12.4),
      "QRS_off": (98.72, -1.7, 14.1), "T_on": (96.35, 8.4, 28.2), "T_off": (96.24, -3.1, 28.2)}, 94.53),
    ("Moskalenko 2020 (5a)", "255 внешних → 200 LUDB", "1 отв.",
     {"P_on": (96.47, 0.9, 14.1), "P_off": (96.43, -3.5, 15.7), "QRS_on": (99.90, 2.1, 9.8),
      "QRS_off": (99.90, 1.6, 9.8), "T_on": (99.15, 1.3, 20.9), "T_off": (99.21, -0.3, 22.9)}, 98.51),
    ("Joung 2024", "1032 внешних → 200 LUDB", "1 отв. (I, II)",
     {"P_on": (97.27, 7.4, 14.1), "P_off": (97.27, -1.8, 9.9), "QRS_on": (99.48, 6.1, 10.5),
      "QRS_off": (99.78, 2.0, 10.7), "T_on": (99.74, 3.0, 25.2), "T_off": (99.52, 4.5, 24.4)}, 98.84),
    ("Moskalenko 2020 (5c)†", "255 внешних → 200 LUDB", "12 отв., усреднение",
     {"P_on": (97.89, -0.6, 17.5), "P_off": (97.85, -2.4, 18.4), "QRS_on": (99.97, 1.5, 11.1),
      "QRS_off": (99.97, 2.0, 10.6), "T_on": (99.52, 2.9, 23.7), "T_off": (99.61, -2.4, 30.4)}, 99.13),
    ("ECG-CODE 2024†", "LUDB (разбиение не указано)", "12 отв. → общая",
     {"P_on": (96.8, -5.1, 20.6), "P_off": (96.8, 5.1, 17.5), "QRS_on": (99.9, 3.6, 10.5),
      "QRS_off": (99.9, 2.4, 16.4), "T_on": (99.3, -5.5, 32.5), "T_off": (96.1, 5.0, 24.8)}, 98.10),
]


# ---------------------------------------------------------------------------------------------- данные
def run_dirs(exp: str):
    return sorted(Path(d) for d in glob.glob(str(REPO / "experiments" / exp / "fold*_seed*"))
                  if (Path(d) / "metrics.json").is_file())


def pooled(exp: str) -> DelineationMetrics:
    """Пулинг test всех прогонов эксперимента (пороги — с val каждого прогона), граничные точки исключены."""
    m = DelineationMetrics(class_names=NAMES, exclude_boundary_points=True)
    for d in run_dirs(exp):
        th = {k: (v["conf"], v["val_F1"]) for k, v in json.loads((d / "thresholds.json").read_text()).items()}
        for s in json.loads((d / "predictions_test.json").read_text()):
            m.add(apply_thresholds(np.asarray(s["pred"], float).reshape(-1, 4), th, NAMES),
                  np.asarray(s["ref"], float).reshape(-1, 3))
    return m


def per_run(exp: str) -> dict:
    """(фолд, сид) → сводка test прогона (⟨F1⟩₆, ⟨σ⟩₆, F1/σ по классам, mAP)."""
    out = {}
    for d in run_dirs(exp):
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        info = json.loads((d / "run.json").read_text(encoding="utf-8"))
        p, s = m["test"]["points"], m["test"]["summary"]
        row = {"F1": 100 * s["F1_mean6"], "sd": s["sd_mean6_ms"], "mAP50": 100 * m["test_detection"]["mAP@50"],
               "mAP": 100 * m["test_detection"]["mAP"]}
        for c in NAMES:
            row[f"F1_{c}"] = 100 * np.nanmean([p[f"{c}_on"]["F1"], p[f"{c}_off"]["F1"]])
            row[f"sd_{c}"] = np.nanmean([p[f"{c}_on"]["sd_ms"], p[f"{c}_off"]["sd_ms"]])
        out[(info["fold"], info["seed"])] = row
    return out


def paired(a: dict, b: dict, key: str):
    """Среднее и SD парных разностей b − a по общим (фолд, сид); число пар, где b > a."""
    keys = sorted(set(a) & set(b))
    d = np.array([b[k][key] - a[k][key] for k in keys])
    return d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d)


def write_table(name: str, md: str, tex: str):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    (OUT / "tables" / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")
    print(OUT / "tables" / f"{name}.md")


def tex_table(caption: str, label: str, columns: str, header: list, rows: list) -> str:
    esc = lambda s: str(s).replace("%", r"\%").replace("±", r"$\pm$").replace("†", r"$^\dagger$").replace("→", r"$\to$")  # noqa: E731
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def md_table(header: list, rows: list) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ---------------------------------------------------------------------------------------------- таблицы
def table_final_points(results: dict):
    header = ["Режим", "Точка", "Se, %", "PPV, %", "F1, %", "m ± σ, мс"]
    rows = []
    for mode, res in results.items():
        for f in POINTS:
            p = res["points"][f]
            rows.append([LABEL[mode], POINT_LABEL[f], f"{100 * p['Se']:.2f}", f"{100 * p['PPV']:.2f}",
                         f"{100 * p['F1']:.2f}", f"{p['mean_ms']:.1f} ± {p['sd_ms']:.1f}"])
        s = res["summary"]
        rows.append([LABEL[mode], "⟨·⟩₆", "", "", f"{100 * s['F1_mean6']:.2f}", f"⟨σ⟩₆ = {s['sd_mean6_ms']:.1f}"])
    caption = ("Делинеация на LUDB, 5-fold CV по пациентам, 2 сида, пулинг test (4752 отведения), допуск ±150 мс; "
               "S — E4 (final), MC — E7 (mc_full)")
    write_table("T1_final_points", f"**{caption}**\n\n" + md_table(header, rows),
                tex_table(caption, "tab:final", "llrrrr", header, rows))


def table_benchmarks(results: dict):
    header = ["Метод", "Обучение → тест", "Вход"] + [POINT_LABEL[f] for f in POINTS] + ["⟨F1⟩₆"]
    rows = []
    for name, protocol, mode, pts, f1 in LITERATURE:
        rows.append([name, protocol, mode] + [f"{pts[f][0]:.2f} ({pts[f][1]:+.1f} ± {pts[f][2]:.1f})" for f in POINTS]
                    + [f"{f1:.2f}"])
    for mode, res in results.items():
        p = res["points"]
        rows.append([LABEL[mode], "5-fold CV LUDB (144 → 40)", "1 отв." if mode == "S" else "12 отв. → по отв."]
                    + [f"{100 * p[f]['F1']:.2f} ({p[f]['mean_ms']:+.1f} ± {p[f]['sd_ms']:.1f})" for f in POINTS]
                    + [f"{100 * res['summary']['F1_mean6']:.2f}"])
    caption = ("Сравнение с опубликованными методами на LUDB: F1, % (m ± σ, мс), допуск 150 мс. "
               "† — общая по 12 отведениям разметка, протокол не поотведенческий; протоколы обучения различаются (R2 §2.2)")
    write_table("T2_benchmarks", f"**{caption}**\n\n" + md_table(header, rows),
                tex_table(caption, "tab:benchmarks", "lll" + "c" * 7, header, rows))


def table_ablation():
    rows_csv = list(csv.DictReader(open(REPO / "experiments" / "ablation" / "results.csv", encoding="utf-8")))
    by = {}
    for r in rows_csv:
        by.setdefault(r["experiment"], {})[(int(r["fold"]), int(r["seed"]))] = {
            "F1": 100 * float(r["F1_mean6"]), "sd": float(r["sd_mean6_ms"]),
            **{f"F1_{c}": 100 * float(r[f"F1_{c}"]) for c in NAMES}}
    order = [("baseline", "Без аугментаций (E2)", None), ("abl_full", "Full: 16 аугментаций", "baseline")]
    groups = {"N": "шумы (N1–N7)", "T": "время (T1–T3)", "A": "амплитуда (A1–A3)", "M": "маскирование (M1)",
              "X": "склейка (X1–X2)"}
    order += [(f"abl_full_minus_{g}", f"Full − {title}", "abl_full") for g, title in groups.items()]
    header = ["Конфигурация", "прогонов", "⟨F1⟩₆, %", "⟨σ⟩₆, мс", "F1 P / QRS / T, %", "ΔF1, п.п.", "Δσ, мс"]
    rows, deltas = [], {}
    for exp, title, ref in order:
        runs = by[exp]
        mean = lambda k: np.mean([r[k] for r in runs.values()])  # noqa: E731
        d_f1 = d_sd = ""
        if ref is not None:
            f, fs, _, n = paired(by[ref], runs, "F1")
            s, ss, _, _ = paired(by[ref], runs, "sd")
            d_f1, d_sd = f"{f:+.2f} ± {fs:.2f}", f"{s:+.2f} ± {ss:.2f}"
            deltas[exp] = (title, f, fs, s, ss, n)
        rows.append([title, len(runs), f"{mean('F1'):.2f}", f"{mean('sd'):.2f}",
                     " / ".join(f"{mean(f'F1_{c}'):.2f}" for c in NAMES), d_f1, d_sd])
    caption = ("Абляция аугментаций (E3): test 5 фолдов, seed 0; Δ — парные разности по фолдам (Full − baseline; "
               "Full − группа − Full); пороги значимости δ_F1 = 0.37 п.п., δ_σ = 0.5 мс")
    write_table("T3_ablation", f"**{caption}**\n\n" + md_table(header, rows),
                tex_table(caption, "tab:ablation", "lrrrrrr", header, rows))
    return deltas


def table_modes():
    runs = {e: per_run(e) for e in ("baseline", "mc_baseline", "final", "mc_full")}
    titles = {"baseline": "S, без аугм. (E2)", "mc_baseline": "MC, без аугм. (E7)", "final": "S, все аугм. (E4)",
              "mc_full": "MC, все аугм. (E7)"}
    header = ["Конфигурация", "⟨F1⟩₆, %", "⟨σ⟩₆, мс", "mAP@50, %", "mAP@50:95, %", "ΔF1 MC − S, п.п.", "Δσ MC − S, мс"]
    rows = []
    for e, title in titles.items():
        r = runs[e].values()
        f = lambda k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r], ddof=1))  # noqa: E731
        d_f1 = d_sd = ""
        ref = {"mc_baseline": "baseline", "mc_full": "final"}.get(e)
        if ref:
            a, sa, pa, n = paired(runs[ref], runs[e], "F1")
            b, sb, pb, _ = paired(runs[ref], runs[e], "sd")
            d_f1, d_sd = f"{a:+.2f} ± {sa:.2f} ({pa}/{n})", f"{b:+.2f} ± {sb:.2f} ({pb}/{n})"
        rows.append([title] + [f"{f(k)[0]:.2f} ± {f(k)[1]:.2f}" for k in ("F1", "sd")]
                    + [f"{f(k)[0]:.1f}" for k in ("mAP50", "mAP")] + [d_f1, d_sd])
    caption = ("Режимы S (1 отведение) и MC (12 отведений → 12 выходов): test 5 фолдов × 2 сида, mean ± SD по прогонам; "
               "Δ — парные разности по (фолд, сид), в скобках — пар с MC > S")
    write_table("T4_modes", f"**{caption}**\n\n" + md_table(header, rows),
                tex_table(caption, "tab:modes", "lrrrrrr", header, rows))


# ---------------------------------------------------------------------------------------------- графики
def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=1)
    ax.set_axisbelow(True)


def save(fig, name):
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / "figures" / f"{name}.png")


def figure_errors(metrics: dict):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharey=False)
    bins = np.arange(-80, 82, 4)
    for ax, f in zip(axes.T.flatten(), POINTS):
        sd = {}
        for mode, m in metrics.items():
            err = np.asarray(m._counts[None][f].errors, float)
            sd[mode] = err.std()
            ax.hist(err, bins=bins, histtype="step", lw=2, color=COLOR[mode], density=True, label=LABEL[mode])
        ax.set_title(POINT_LABEL[f], fontsize=10, color=INK, loc="left")
        ax.text(0.0, 1.0, "σ: " + " · ".join(f"{m} {v:.1f}" for m, v in sd.items()) + " мс", transform=ax.transAxes,
                fontsize=8, color=INK2, va="top")
        ax.set_xlim(-80, 80)
        ax.set_yticks([])
        style(ax, "x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("ошибка положения (pred − эксперт), мс", fontsize=8, color=INK2)
    fig.suptitle("Распределение ошибки по 6 точкам (test, 5 фолдов × 2 сида; совпавшие точки, допуск ±150 мс)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F1_error_distributions")


def figure_learning_curves():
    steps_per_epoch = {"S": 54, "MC": 18}
    panels = [("F1_mean6", "val ⟨F1⟩₆, %", 100, (0.90, 1.0)), ("sd_mean6_ms", "val ⟨σ⟩₆, мс", 1, (8, 30)),
              ("val_loss", "val лосс (box + cls + dfl)", 1, None)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, title, scale, lim) in zip(axes, panels):
        for mode, exp in EXPERIMENTS.items():
            curves = []
            for d in run_dirs(exp):
                rows = list(csv.DictReader(open(d / "train1" / "step.csv")))
                if key == "val_loss":
                    y = [float(r["box_val"]) + float(r["cls_val"]) + float(r["dfl_val"]) for r in rows]
                else:
                    y = [float(r[key]) if r[key] not in ("nan", "") else np.nan for r in rows]
                curves.append(np.asarray(y) * scale)
                ax.plot(np.arange(1, len(y) + 1) * steps_per_epoch[mode] / 1000, np.asarray(y) * scale,
                        color=COLOR[mode], lw=0.6, alpha=0.25)
            n = max(len(c) for c in curves)
            stack = np.full((len(curves), n), np.nan)
            for i, c in enumerate(curves):
                stack[i, :len(c)] = c
            alive = (~np.isnan(stack)).sum(0) >= len(curves) / 2  # медиана, пока живо ≥ половины прогонов
            x = np.arange(1, n + 1) * steps_per_epoch[mode] / 1000
            ax.plot(x[alive], np.nanmedian(stack, 0)[alive], color=COLOR[mode], lw=2, label=LABEL[mode])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("шаги оптимизатора, тыс.", fontsize=8, color=INK2)
        if lim:
            ax.set_ylim(lim[0] * (100 if scale == 100 else 1), lim[1] * (100 if scale == 100 else 1))
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("Кривые обучения (тонкие линии — 10 прогонов, толстые — медиана): S (E4) и MC (E7), все аугментации",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F2_learning_curves")


def figure_ablation(deltas: dict):
    names = {"N": "без шумов (N1–N7)", "T": "без временных (T1–T3)", "A": "без амплитудных (A1–A3)",
             "M": "без маскирования (M1)", "X": "без склейки (X1–X2)"}
    items = [(names[k.rsplit("_", 1)[-1]], *v[1:]) for k, v in deltas.items() if k != "abl_full"]
    full = deltas["abl_full"]
    labels = ["Full против\nбез аугм."] + [i[0] for i in items]
    f1 = [full[1]] + [i[1] for i in items]
    f1s = [full[2]] + [i[2] for i in items]
    sd = [full[3]] + [i[3] for i in items]
    sds = [full[4]] + [i[4] for i in items]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, val, err, thr, title in ((axes[0], f1, f1s, 0.37, "Δ⟨F1⟩₆, п.п."), (axes[1], sd, sds, 0.5, "Δ⟨σ⟩₆, мс")):
        ax.axvspan(-thr, thr, color="#f0efec", lw=0)
        ax.axvline(0, color=INK2, lw=1)
        ax.barh(y, val, height=0.45, color=["#52514e"] + [COLOR["S"]] * len(items))
        ax.errorbar(val, y, xerr=err, fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax, "x")
    axes[0].set_yticks(y, labels, fontsize=8, color=INK)
    fig.suptitle("Абляция аугментаций (E3): парные разности по 5 фолдам (mean ± SD); серая полоса — порог шума "
                 "(δ_F1 0.37 п.п., δ_σ 0.5 мс). Строки «без группы» — относительно Full",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F3_ablation")


def figure_examples():
    sys.path.insert(0, str(REPO / "experiments" / "scripts"))
    import plot_predictions as pp
    from plot_s_vs_mc import MCModel
    from yolo_1d.utils.ludb import LEADS
    from yolo_1d.utils.multilead_dataset import MultiLeadDataset
    from yolo_1d.utils.transforms import EdgeMask
    cases = [("151", "ii", "синусовый ритм"), ("116", "v2", "полная блокада ножки"), ("103", "ii", "трепетание предсердий")]
    with open(REPO / "data" / "splits" / "folds.csv", encoding="utf-8") as f:
        fold_of = {r["record"]: int(r["fold"]) for r in csv.DictReader(f)}
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 2.6 * len(cases)))
    rows = {"эксперт": 1.0, "S": 0.0, "MC": -1.0}
    for ax, (rec, lead, caption) in zip(axes, cases):
        k = fold_of[rec]
        pp.FOLD_YAML = REPO / "data" / "splits" / f"fold_{k}.yaml"
        s_model = pp.Model(REPO / "experiments" / "final" / f"fold{k}_seed2")
        mc_yaml = REPO / "data" / "splits" / f"fold_{k}_mc.yaml"
        mc_model = MCModel(REPO / "experiments" / "mc_full" / f"fold{k}_seed2", mc_yaml)
        ds = MultiLeadDataset(str(mc_yaml), None, "test", edge_mask=EdgeMask((6, 6), (12, 12)))
        i = [ds.sample_id(j) for j in range(ds.n)].index(rec)
        li = LEADS.index(lead)
        x = ds.load_signal(i)
        lab = ds.lead_labels(i, li)
        ref = np.c_[lab[:, 0], (lab[:, 1] - lab[:, 2] / 2) * 5000, (lab[:, 1] + lab[:, 2] / 2) * 5000]
        sig = x[li].numpy()
        sig = sig / max(np.abs(sig).max(), 1e-6)
        nz = np.nonzero(sig)[0]
        t = np.arange(len(sig)) / 500
        ax.plot(t, sig * 1.2 + 2.6, color=INK, lw=0.8)
        for name, intervals in (("эксперт", ref), ("S", s_model.predict(x[li:li + 1])), ("MC", mc_model.predict(x)[li])):
            for c, l, r, *_ in intervals:
                ax.barh(rows[name], (r - l) / 500, left=l / 500, height=0.6, color=COLOR[NAMES[int(c)]],
                        edgecolor="white", lw=1)
        ax.set_yticks(list(rows.values()), list(rows.keys()), fontsize=8, color=INK)
        ax.set_xlim(nz[0] / 500 - 0.1, nz[-1] / 500 + 0.1)
        ax.set_ylim(-1.6, 4.2)
        ax.set_title(f"Запись {rec}, отведение {lead.upper()} — {caption}", fontsize=9, color=INK, loc="left")
        style(ax, "x")
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("время, с", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in NAMES]
    fig.legend(handles, NAMES, ncol=3, frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    fig.suptitle("Примеры делинеации (test): разметка эксперта и ответы MyNet-1d в режимах S и MC",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F4_examples")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-examples", action="store_true", help="не строить F4 (нужны GPU/чекпоинты)")
    args = parser.parse_args()
    metrics = {mode: pooled(exp) for mode, exp in EXPERIMENTS.items()}
    results = {mode: m.compute() for mode, m in metrics.items()}
    table_final_points(results)
    table_benchmarks(results)
    deltas = table_ablation()
    table_modes()
    figure_errors(metrics)
    figure_learning_curves()
    figure_ablation(deltas)
    if not args.skip_examples:
        figure_examples()
    return 0


if __name__ == "__main__":
    sys.exit(main())
