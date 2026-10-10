# Многоканальные ЭКГ-аугментации режима MC (A3)

Реализация — [ecg_multilead_transforms.py](../../../src/yolo_1d/utils/transforms/ecg_multilead_transforms.py), склейка X1/X2 — `MultiLeadDataset.load_splice` ([multilead_dataset.py](../../../src/yolo_1d/utils/multilead_dataset.py)), сборка — `build_ecg_multilead_training_transforms` и `build_lead_transforms`. Включение: `augmentations.pipeline: ecg_multi`; вероятности сигнальных аугментаций — те же поля `augmentations.ecg.*`, что в режиме S, группа L — `augmentations.leads.*`. Обоснование — [R5](../../research/R5_multilead.md) §3.
Картинки: запись 1 из train фолда 0, слева — исходные 12 отведений, справа — после аугментации с p = 1 (P — зелёный, QRS — красный, T — синий; серым — отведения, исключённые из лосса). Генерация: `python experiments/scripts/plot_multilead_augmentations.py`.

| ID | Класс | В MC | Параметры | Картинка |
|---|---|---|---|---|
| N1 | `CoupledNoise("wander")` | общий источник, коэффициент отведения U(0.5, 1.5)·(±1) | `ecg.wander_*` | [N1](N1_wander.png) |
| N2 | `CoupledNoise("powerline")` | то же | `ecg.powerline_*` | — |
| N3 | `EMGBursts` (A1) | шум независим по отведениям, окна общие | `ecg.emg_*` | — |
| N4 | `WhiteNoise` (A1) | независимо по отведениям | `ecg.white_*` | — |
| N5 | `CoupledNoise("nstdb")` | общий фрагмент NSTDB, коэффициент отведения | `ecg.nstdb_*` | [N5](N5_nstdb.png) |
| N6 | `LeadBaselineStep` | ступенька в 1–3 отведениях | `ecg.step_*` | [N6](N6_step.png) |
| N7 | `RandomFilter` (A1) | один фильтр для всех отведений | `ecg.filter_p` | — |
| T1, T3 | `TimeScale`, `WindowWarp` (A1) | одно отображение времени | `ecg.time_scale_*`, `ecg.window_warp_*` | — |
| T2 | `MultiLeadHRVWarp` | узлы — по кластерам волн всех отведений | `ecg.hrv_*` | [T2](T2_hrv_warp.png) |
| A1 | `MultiLeadAmplitudeScale` | общий × поканальный lognormal | `ecg.amp_scale_*`, `leads.amp_lead_sigma` | [A1](A1_amplitude.png) |
| A2 | `MultiLeadWaveScale` | общий множитель волны сердца × поканальный, в своём отведении | `ecg.wave_scale_*`, `leads.amp_lead_sigma` | [A2](A2_wave_scale.png) |
| A3 | — | **не используется** (поканальная инверсия нефизиологична) | `ecg.invert_p` игнорируется | — |
| M1 | `TimeMask` (A1) | общий отрезок | `ecg.time_mask_*` | [M1](M1_time_mask.png) |
| X1, X2 | `MultiLeadDataset.load_splice` | общий разрез; X2 — в паузе после T во всех отведениях | `ecg.mosaic_p`, `ecg.splice_p` | [X2](X2_beat_splice.png) |
| L1 | `LeadDropout` | обнуление ≤ 4 отведений; разметка `ignore` / `keep` | `leads.dropout_*` | [L1](L1_lead_dropout.png) |
| L2 | `LeadShuffle` | перестановка 2–4 отведений с разметкой | `leads.shuffle_*` | [L2](L2_lead_shuffle.png) |
| L3 | `ElectrodeSwap` | LA↔RA, LA↔LL, RA↔LL (точные формулы) | `leads.electrode_swap_p` | [L3](L3_electrode_LA_RA.png) |

Полный пайплайн сигнальных аугментаций (вероятности E4): [пример 1](full_pipeline_1.png), [пример 2](full_pipeline_2.png).

Общие правила: шумы и фильтр обёрнуты в `KeepZeros` — нули маски краёв каждого отведения сохраняются (как на валидации); метки волн внутри трансформаций — (n, 2) [class, lead], так что отбор и пересчёт волн сохраняют отведение; группа L работает после сигнальных аугментаций и меняет маску `valid`. Порядок: время → амплитуда → шумы → фильтр → маска времени → `FitToLength` → L (перестановка электродов → перестановка отведений → dropout).
