"""Тесты выбора подмножества входных каналов (input_subset, абляция входа E6, задача D8)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from test_ecr_pipeline import FULL, OFF, INP, make_ecr, synthetic, run  # noqa: E402
from yolo_1d.nets import nn  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402
from yolo_1d.utils.transforms import (EddyPreprocessing, SelectChannels, build_ecr_training_transforms,  # noqa: E402
                                      build_transforms_for_dataset, build_validation_transform)


@pytest.mark.parametrize("subset,channels", [("all", 30), ("raw", 15), ("filtered", 15)])
def test_settings_input_channels(subset, channels):
    s = TrainingSettings(names=["a"], input_subset=subset)
    assert s.input_channels == channels


def test_settings_conflict_and_ecg_untouched():
    with pytest.raises(ValidationError):
        TrainingSettings(names=["a"], input_subset="raw", input_channels=30)
    assert TrainingSettings(names=["a"], input_channels=1).input_channels == 1      # ЭКГ-конфиги


def test_select_channels():
    x = torch.arange(30.0)[:, None].expand(30, 5)
    assert torch.equal(SelectChannels("raw")(x, None, None)[0], x[:15])
    assert torch.equal(SelectChannels("filtered")(x, None, None)[0], x[15:])
    assert torch.equal(SelectChannels("all")(x, None, None)[0], x)
    with pytest.raises(ValueError):
        SelectChannels("both")


@pytest.mark.parametrize("subset,half", [("raw", slice(0, 15)), ("filtered", slice(15, 30))])
def test_subset_equals_half_of_full_output(subset, half):
    s = AugmentationSettings(**OFF)
    x = synthetic(seed=2)
    full, _, _ = run(build_validation_transform(s, EddyPreprocessing()), x)
    part, _, _ = run(build_validation_transform(s, EddyPreprocessing(), input_subset=subset), x)
    torch.testing.assert_close(part, full[half], rtol=0, atol=0)
    tr, _, _ = run(build_ecr_training_transforms(s, EddyPreprocessing(), subset), x)
    torch.testing.assert_close(tr, full[half], rtol=0, atol=0)      # все аугментации выключены → train = val


def test_select_before_channel_augmentations():
    names = [type(getattr(op, "augmentation", op)).__name__ for op in
             build_ecr_training_transforms(AugmentationSettings(**FULL), EddyPreprocessing(), "raw").transforms]
    i = names.index("SelectChannels")
    assert names[i - 1] == "ThresholdNormalize" and names[i + 1:] == ["ChannelScaling", "ChannelMasking", "RandomPadding"]


def test_subset_requires_ecr(tmp_path):
    with pytest.raises(ValueError, match="ECRv1"):
        build_transforms_for_dataset(AugmentationSettings(), None, input_subset="raw")


@pytest.mark.parametrize("subset,channels", [("all", 30), ("raw", 15), ("filtered", 15)])
def test_batch_shape_and_model(tmp_path, subset, channels):
    m = make_ecr(tmp_path)
    s = AugmentationSettings(**FULL)
    tr, te = build_transforms_for_dataset(s, str(m), input_subset=subset)
    for ds in (Dataset(str(m), tr, "train", mosaic_prob=s.mosaic, mix_up_prob=s.mix_up), Dataset(str(m), te, "val")):
        x, t = next(iter(DataLoader(ds, batch_size=2, collate_fn=Dataset.collate_fn)))
        assert x.shape == (2, channels, INP) and torch.isfinite(x).all()
    settings = TrainingSettings(names=["a"] * 6, input_subset=subset)
    model = nn.get_model("nano", 6, input_channels=settings.input_channels, input_length=INP)
    with torch.no_grad():
        model.eval()
        model(x)
