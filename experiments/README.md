# experiments/

Эксперименты MyNet-1d на LUDB запускаются по YAML-конфигам из `configs/`:

```bash
python run_experiment.py experiments/configs/ludb_baseline_A.yaml               # все фолды и сиды из конфига
python run_experiment.py experiments/configs/ludb_baseline_A.yaml --folds 0 --epochs 3   # быстрый прогон
python run_experiment.py experiments/configs/ludb_baseline_A.yaml --summary-only         # пересобрать сводку
```

Структура результата `experiments/<name>/`:

| Файл | Что |
|---|---|
| `config.yaml`, `environment.json` | конфиг с переопределениями CLI; git hash/ветка/«грязная» рабочая копия, версии Python/torch/CUDA, GPU, команда |
| `fold<k>_seed<s>/train<n>/step.csv` | метрики по эпохам: лоссы, mAP/P/R, `F1_mean6`, `sd_mean6_ms`, F1 каждой точки |
| `fold<k>_seed<s>/train<n>/args.yaml` | все гиперпараметры прогона (`TrainingSettings`) |
| `fold<k>_seed<s>/metrics.json`, `metrics.md` | итог на test: пороги по классам (подобраны на val), Se/PPV/F1/m±σ по 6 точкам, по группам ритма, mAP |
| `fold<k>_seed<s>/thresholds.json`, `run.json` | пороги уверенности, длительность, использованный чекпоинт |
| `summary.json`, `summary.md` | все прогоны, mean ± SD по фолдам, **пулинг тестов всех фолдов** (итоговая таблица, R2 §3.2) |
| `results.csv` (в корне `experiments/`) | реестр: одна строка на эксперимент |

В git не хранятся чекпоинты (`*.pt`) и сырые предсказания (`predictions_*.json`): они остаются локально и позволяют пересчитать сводку (`--summary-only`).

Протокол метрик — `docs/research/R2_metrics_benchmarks.md` §3; основная метрика выбора модели и абляции — `F1_mean6`.
