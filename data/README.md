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

### Подготовка для обучения

```bash
python data/prepare_ludb.py    # -> data/processed/ludb/ и data/ludb.yaml
```

**Разметка на отведения — решение (2026-10-07): детекция поотведенческая.** Пример = одно отведение одной записи, вход сети 1 × 5000 (паддинг до 5120 в трансформациях), разметка — собственная разметка этого отведения. 12-канальный режим с агрегированной разметкой не используется.

Что делает скрипт (подробнее — docstring `data/prepare_ludb.py`, `src/yolo_1d/utils/ludb_preprocess.py`):
1. Z-score по всему отведению (полосовой фильтр — опция `--bandpass`, по умолчанию выключен).
2. Сигнал вне размеченной зоны `[первый onset − 10, последний offset + 10]` отсчётов заполняется нулями (`--fill median` — медианой отведения). Неразмеченные волны на краях в сеть не попадают; вход одинаков на обучении и валидации.
3. Разметка: строки `class center width`, нормированные на 5000; left = onset, right = offset; P = 0, QRS = 1, T = 2.
4. 24 отведения, где внутри зоны есть волна без метки (дефект разметки, см. отчёт D2), исключаются из всех выборок (`--keep-defective` — оставить).

Выход `data/processed/ludb/`: `signals/<запись>_<отведение>.pt`, `labels/<запись>_<отведение>.txt`, `manifest.csv` (все 2400 отведений: зона, число волн, группа ритма, `usable`, причина исключения), `preprocessing.json` (параметры). Конфиг `data/ludb.yaml` содержит классы и пути; разбиения добавляет D4.

Лицензия: Open Data Commons Attribution v1.0. Цитировать: Kalyakulina A. I. et al., *LUDB: A New Open-Access Validation Tool for Electrocardiogram Delineation Algorithms*, IEEE Access 8, 2020, и PhysioNet (doi:10.13026/eegm-h675). Особенности разметки описаны в [docs/research/R1_ludb.md](../docs/research/R1_ludb.md).
