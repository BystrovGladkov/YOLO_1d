# data/

Датасеты и всё, что с ними связано. Сырые данные в git не хранятся (`data/raw/` в `.gitignore`), их нужно скачать скриптом.

## LUDB

```bash
python data/download_ludb.py               # скачать в data/raw/ludb/ и проверить
python data/download_ludb.py --check-only  # проверить уже скачанное
```

Скрипт скачивает LUDB v1.0.1 с PhysioNet (13.8 МБ) и проверяет:
- SHA-256 всех файлов по `SHA256SUMS.txt`;
- 200 записей × 12 отведений × 5000 отсчётов @ 500 Гц;
- наличие и читаемость разметки каждого отведения.

Структура `data/raw/ludb/`: `RECORDS`, `ludb.csv` (метаданные: ритм, диагнозы, ЭКС), `data/<N>.hea|.dat` (сигнал) и `data/<N>.<lead>` (разметка отведения; `lead` ∈ i, ii, iii, avr, avl, avf, v1–v6).

Чтение в коде — модуль `yolo_1d.utils.ludb` (`load_record`, `iter_records`): сигнал (12, 5000) в мВ, волны P/QRS/T по отведениям, дефекты разметки, метаданные из `ludb.csv`.

Отчёт о данных (сверка с публикацией, статистика, рисунки) → `data/reports/`:

```bash
python data/ludb_report.py
```

Лицензия: Open Data Commons Attribution v1.0. Цитировать: Kalyakulina A. I. et al., *LUDB: A New Open-Access Validation Tool for Electrocardiogram Delineation Algorithms*, IEEE Access 8, 2020, и PhysioNet (doi:10.13026/eegm-h675). Особенности разметки описаны в [docs/research/R1_ludb.md](../docs/research/R1_ludb.md).
