from torchvision.transforms.v2 import Compose

from .transforms import Padding, Resize, RandomResize, RandomCrop, RandomPadding, RandomReverse, RandomJitter, RandomInvert, TimeMasking, ChannelMasking, ChannelScaling, MarkScaling, PhaseSpectrumNoise, RandomApply
from yolo_1d.utils.settings import AugmentationSettings
from .square_noises import SquareNoises
from .ecg_transforms import (AmplitudeScale, BaselineStep, BaselineWander, EMGBursts, FitToLength, HRVWarp,
                             Invert, NSTDBNoise, PowerlineNoise, RandomFilter, TimeMask, TimeScale, WaveScale,
                             WhiteNoise, WindowWarp)

def build_ecg_training_transforms(settings: AugmentationSettings):
    """Аугментации обучения для ЭКГ (A1). Порядок — docs/research/R3_augmentations.md §4.1:
    время (T2, T3, T1) → амплитуда (A2, A1, A3) → шумы (N1, N6, N5, N3, N2, N4) → фильтрация (N7)
    → маскирование (M1) → приведение к длине входа со случайным сдвигом. Маска краёв (EdgeMask)
    применяется раньше, при загрузке; склейка записей X1/X2 — в Dataset."""
    e = settings.ecg
    ops = [
        HRVWarp(e.hrv_warp_p, tp=e.hrv_tp),
        WindowWarp(e.window_warp_p, scale=e.window_warp_scale),
        TimeScale(e.time_scale_p, scale=e.time_scale),
        WaveScale(e.wave_scale_p, sigma_pt=e.wave_scale_sigma_pt),
        AmplitudeScale(e.amp_scale_p, sigma=e.amp_scale_sigma),
        Invert(e.invert_p),
        BaselineWander(e.wander_p, max_amp=e.wander_amp, freq=e.wander_freq),
        BaselineStep(e.step_p, max_amp=e.step_amp),
        NSTDBNoise(e.nstdb_p, root=e.nstdb_root, snr=e.nstdb_snr),
        EMGBursts(e.emg_p, snr=e.emg_snr),
        PowerlineNoise(e.powerline_p, snr=e.powerline_snr),
        WhiteNoise(e.white_p, snr=e.white_snr),
        RandomFilter(e.filter_p),
        TimeMask(e.time_mask_p, max_fraction=e.time_mask_max),
    ]
    ops = [op for op in ops if op.p > 0]
    ops.append(FitToLength(settings.input_length, random_shift=True))
    return Compose(ops)


def build_training_transforms(settings: AugmentationSettings):
    if settings.pipeline == "ecg":
        return build_ecg_training_transforms(settings)
    ops = [
        # MarkScaling(settings.mark_scaling),
        RandomResize(settings.random_resize),
        RandomCrop(settings.input_length),
    ]

    if settings.square_noises > 0:
        ops.append(RandomApply(SquareNoises(), p=settings.square_noises))
    
    if settings.phase_spectrum_noise > 0:
        ops.append(PhaseSpectrumNoise(settings.phase_spectrum_noise))
    if settings.random_jitter > 0:
        ops.append(RandomJitter(settings.random_jitter))
    if settings.random_invert > 0:
        ops.append(RandomInvert(settings.random_invert))
    if settings.random_reverse > 0:
        ops.append(RandomReverse(settings.random_reverse))
    if settings.channel_scaling > 0:
        ops.append(ChannelScaling())
    if settings.time_masking > 0:
        ops.append(TimeMasking(settings.time_masking))
    if settings.channel_masking > 0:
        ops.append(ChannelMasking(settings.channel_masking))
    
    ops.append(RandomPadding(settings.input_length))

    return Compose(ops)

def build_validation_transform(settings: AugmentationSettings):
    ops = []
    if settings.validation_resize:
        # вихретоковые данные: паддинг до regular_length и ресайз до input_length
        ops.append(Padding(settings.regular_length))
        ops.append(Resize(settings.input_length))
    else:
        # ЭКГ: только паддинг до input_length, временной масштаб не меняется
        ops.append(Padding(settings.input_length))
    return Compose(ops)