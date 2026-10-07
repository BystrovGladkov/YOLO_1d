# ЭКГ-аугментации (A1)

Реализация — [src/yolo_1d/utils/transforms/ecg_transforms.py](../../src/yolo_1d/utils/transforms/ecg_transforms.py) (X1/X2 — `Dataset.load_splice` в [dataset.py](../../src/yolo_1d/utils/dataset.py)), сборка — `build_ecg_training_transforms`. Включение: `augmentations.pipeline: ecg`; вероятности и параметры — `augmentations.ecg.*` (`ECGAugmentationSettings`, при `*_p = 0` аугментация выключена). Обоснование и диапазоны — [R3](../research/R3_augmentations.md) §3.
Картинки: по 3 отведения LUDB из train фолда 0, сверху исходный сигнал, снизу — после аугментации с p = 1 (P — зелёный, QRS — красный, T — синий). Генерация: `python experiments/scripts/plot_augmentations.py`.

| ID | Класс | Параметры конфига (`augmentations.ecg`) | Картинка |
|---|---|---|---|
| N1 | `BaselineWander` | `wander_p`, `wander_amp` (0.5 SD), `wander_freq` (0.05–0.5 Гц) | [N1](N1_baseline_wander.png) |
| N2 | `PowerlineNoise` | `powerline_p`, `powerline_snr` (15–35 дБ) | [N2](N2_powerline.png) |
| N3 | `EMGBursts` | `emg_p`, `emg_snr` (10–25 дБ) | [N3](N3_emg_bursts.png) |
| N4 | `WhiteNoise` | `white_p`, `white_snr` (20–35 дБ) | [N4](N4_white_noise.png) |
| N5 | `NSTDBNoise` | `nstdb_p`, `nstdb_snr` (6–24 дБ; `em` ≥ 12 дБ), `nstdb_root` | [N5](N5_nstdb.png) |
| N6 | `BaselineStep` | `step_p`, `step_amp` (1 SD) | [N6](N6_baseline_step.png) |
| N7 | `RandomFilter` | `filter_p` (ФВЧ 0.05–0.7 Гц и/или ФНЧ 25–100 Гц, нулевая фаза) | [N7](N7_random_filter.png) |
| T1 | `TimeScale` | `time_scale_p`, `time_scale` (0.85–1.15) | [T1](T1_time_scale.png) |
| T2 | `HRVWarp` | `hrv_warp_p`, `hrv_tp` (диастола 0.6–1.5; PR/ST 0.85–1.15; волны 0.95–1.05) | [T2](T2_hrv_warp.png) |
| T3 | `WindowWarp` | `window_warp_p`, `window_warp_scale` (0.7–1.4, окно 10–30 %) | [T3](T3_window_warp.png) |
| A1 | `AmplitudeScale` | `amp_scale_p`, `amp_scale_sigma` (lognormal 0.2) | [A1](A1_amplitude_scale.png) |
| A2 | `WaveScale` | `wave_scale_p`, `wave_scale_sigma_pt` (0.3; QRS 0.1) | [A2](A2_wave_scale.png) |
| A3 | `Invert` | `invert_p` | [A3](A3_invert.png) |
| M1 | `TimeMask` | `time_mask_p`, `time_mask_max` (10 %) | [M1](M1_time_mask.png) |
| X1 | `Dataset.load_splice(aligned=False)` | `mosaic_p` | [X1](X1_mosaic.png) |
| X2 | `Dataset.load_splice(aligned=True)` | `splice_p` | [X2](X2_beat_splice.png) |

Общие правила: шумы и фильтрация действуют только в активной части сигнала (обнулённые маской края остаются нулями, как на валидации); временные преобразования пересчитывают боксы тем же отображением времени; последний шаг `FitToLength` приводит сигнал к 5120 со случайным сдвигом, не разрезая волны. Порядок: время → амплитуда → шумы → фильтрация → маскирование → `FitToLength`.
