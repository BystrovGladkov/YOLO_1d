# Материалы для статьи (P1)

Генерация: `python experiments/report/make_report.py` (F4 загружает чекпоинты и строит предсказания; без него — `--skip-examples`). Все числа пересчитываются из логов экспериментов: `predictions_test.json` + `thresholds.json` прогонов, `step.csv`, `experiments/ablation/results.csv`; литературные значения — [R2](../../docs/research/R2_metrics_benchmarks.md) §2.1.

| Файл | Содержание | Источник |
|---|---|---|
| [tables/T1_final_points](tables/T1_final_points.md) (.md, .tex) | Se / PPV / F1 / m ± σ по 6 точкам и ⟨·⟩₆: режим S (E4) и MC (E7) | `experiments/final`, `experiments/mc_full` |
| [tables/T2_benchmarks](tables/T2_benchmarks.md) | сравнение с Sereda 2019, Moskalenko 2020 (5a, 5c), Joung 2024, ECG-CODE 2024 | R2 + E4/E7 |
| [tables/T3_ablation](tables/T3_ablation.md) | абляция аугментаций: baseline, Full, Full − группа; парные разности | E2, E3 |
| [tables/T4_modes](tables/T4_modes.md) | режимы S и MC без / со всеми аугментациями, mAP, парные разности MC − S | E2, E4, E7 |
| [figures/F1_error_distributions.png](figures/F1_error_distributions.png) | распределение ошибки (pred − эксперт) по 6 точкам, S и MC | E4, E7 |
| [figures/F2_learning_curves.png](figures/F2_learning_curves.png) | val ⟨F1⟩₆, val ⟨σ⟩₆, val-лосс по шагам оптимизатора, 10 прогонов + медиана | E4, E7 |
| [figures/F3_ablation.png](figures/F3_ablation.png) | парные ΔF1 и Δσ по группам аугментаций с порогами шума | E2, E3 |
| [figures/F4_examples.png](figures/F4_examples.png) | примеры делинеации (записи 151, 116, 103): эксперт, S, MC | E4, E7 (seed 2) |

Соглашения:
- метрики — test, пулинг всех прогонов (5 фолдов × 2 сида), допуск ±150 мс, граничные точки исключены;
- SD — **выборочное (n − 1)**; в `summary.md` прогонов и в `experiments/ablation/summary.md` SD считалось по n, поэтому SD в T3/T4 немного больше (например, 0.19 против 0.17);
- LaTeX-таблицы содержат Unicode (кириллица, ⟨⟩, σ, ₆) — компилировать XeLaTeX/LuaLaTeX (`fontspec`) с пакетом `booktabs`;
- цвета графиков — проверенная палитра (dataviz `validate_palette.js`: S/T — синий, MC/QRS — оранжевый, P — бирюзовый; все пары различимы при дальтонизме), подписи и легенды — нейтральным цветом.
