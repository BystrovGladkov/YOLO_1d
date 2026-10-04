from torchvision.transforms.v2 import Compose

from .transforms import Padding, Resize, RandomResize, RandomCrop, RandomPadding, RandomReverse, RandomJitter, RandomInvert, TimeMasking, ChannelMasking, ChannelScaling, MarkScaling, PhaseSpectrumNoise, RandomApply
from yolo_1d.utils.settings import AugmentationSettings
from .square_noises import SquareNoises

def build_training_transforms(settings: AugmentationSettings):
    ops = [
        # MarkScaling(settings.mark_scaling),
        RandomResize(settings.random_resize),
        RandomCrop(settings.input_length),
    ]

    ops.append(RandomApply(SquareNoises(), p=0.1))
    
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
    ops.append(Padding(settings.regular_length))
    ops.append(Resize(settings.input_length))
    return Compose(ops)