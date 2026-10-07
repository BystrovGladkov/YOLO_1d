"""Тесты M1: длина входа и число каналов — параметры, исходная (вихретоковая) конфигурация не изменилась."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.nets import nn  # noqa: E402
from yolo_1d.nets.export import ExportWrapper  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings, TrainingSettings  # noqa: E402
from yolo_1d.utils.transforms import build_training_transforms, build_validation_transform  # noqa: E402

FOLD0 = REPO / "data" / "splits" / "fold_0.yaml"
PROCESSED = REPO / "data" / "processed" / "ludb" / "manifest.csv"

# настройки ЭКГ (LUDB): вход = вся запись 5000 + паддинг до 5120, без ресайза, 1 канал
ECG_AUG = dict(regular_length=5000, input_length=5120, validation_resize=False, square_noises=0,
               random_resize=0, phase_spectrum_noise=0, random_jitter=0, random_invert=0, random_reverse=0,
               channel_scaling=0, time_masking=0, channel_masking=0)


def test_defaults_keep_eddy_current_configuration():
    s = TrainingSettings(names=["a", "b"])
    assert s.input_channels == 30
    a = s.augmentations
    assert (a.regular_length, a.input_length, a.validation_resize, a.square_noises) == (50000, 49152, True, 0.1)
    model = nn.get_model("nano", 6)
    assert model.input_channels == 30 and model.input_length == 49152
    assert model.stride.tolist() == [32, 128, 256]


def test_ecg_model_one_channel_5120():
    model = nn.get_model("nano", 3, input_channels=1, input_length=5120)
    assert model.input_length == 5120 and model.head.input_length == 5120
    model.train()
    maps = model(torch.zeros(2, 1, 5120))
    assert [m.shape[-1] for m in maps] == [5120 // int(s) for s in model.stride]
    model.eval()
    out = model(torch.zeros(2, 1, 5120))
    assert out.shape == (2, 2 + 3, sum(5120 // int(s) for s in model.stride))


def test_validation_transform_ecg_only_pads():
    transform = build_validation_transform(AugmentationSettings(**ECG_AUG))
    x = torch.arange(5000, dtype=torch.float32)[None]
    box = torch.tensor([[0.5, 0.1]])
    y, b, _ = transform(x.clone(), box.clone(), torch.zeros(1))
    assert y.shape == (1, 5120)
    assert torch.equal(y[0, :5000], x[0]) and torch.all(y[0, 5000:] == 0)  # масштаб времени не меняется
    assert b[0].tolist() == pytest.approx([0.5 * 5000 / 5120, 0.1 * 5000 / 5120])


def test_validation_transform_eddy_current_unchanged():
    transform = build_validation_transform(AugmentationSettings())
    y, b, _ = transform(torch.zeros(30, 50000), torch.tensor([[0.5, 0.1]]), torch.zeros(1))
    assert y.shape == (30, 49152)
    assert b[0].tolist() == pytest.approx([0.5, 0.1])


def test_training_transform_ecg_keeps_boxes_on_signal():
    """Без аугментаций: паддинг до 5120 со случайным сдвигом, боксы указывают на те же отсчёты."""
    transform = build_training_transforms(AugmentationSettings(**ECG_AUG))
    x = torch.zeros(1, 5000)
    x[0, 1000:1200] = 1.0  # «волна» 1000…1200
    box = torch.tensor([[1100 / 5000, 200 / 5000]])
    for _ in range(5):
        y, b, _ = transform(x.clone(), box.clone(), torch.zeros(1))
        assert y.shape == (1, 5120)
        left, right = (b[0, 0] - b[0, 1] / 2) * 5120, (b[0, 0] + b[0, 1] / 2) * 5120
        ones = torch.nonzero(y[0]).flatten()
        assert ones.min().item() == pytest.approx(left.item(), abs=1)
        assert ones.max().item() + 1 == pytest.approx(right.item(), abs=1)


def test_export_wrapper_pads_without_resize():
    model = nn.get_model("nano", 3, input_channels=1, input_length=5120)
    wrapper = ExportWrapper(model, input_length=5120, resize=False)
    with torch.no_grad():
        out = wrapper(torch.zeros(1, 1, 5000))
    assert out.shape[-1] == sum(5120 // int(s) for s in model.stride)


@pytest.mark.skipif(not FOLD0.is_file() or not PROCESSED.is_file(), reason="нет data/splits или data/processed")
@pytest.mark.parametrize("arch", [{}, {"backbone_strides": "A", "bias_init": "1d"}])
def test_trainer_runs_ecg_batch_through_loss(tmp_path, arch):
    """Сквозная проверка: Trainer с настройками ЭКГ, батч LUDB фолда 0 → модель → функция потерь."""
    from yolo_1d.yolo_1d import Trainer

    settings = TrainingSettings(names=["P", "QRS", "T"], dataset=str(FOLD0), results_path=str(tmp_path),
                                input_channels=1, batch_size=4, num_workers=0,
                                augmentations=AugmentationSettings(**ECG_AUG), **arch)
    trainer = Trainer(settings)
    expected = nn.head_strides(settings.backbone_strides)
    assert tuple(int(s) for s in trainer._model.stride) == expected
    trainer._init_dataloaders()
    trainer._init_training()
    samples, targets = next(iter(trainer._train_loader))
    assert samples.shape == (4, 1, 5120)
    assert targets["box"].shape[0] > 0
    trainer._model.train()
    outputs = trainer._model(samples.to(trainer._device).float())
    losses = trainer._criterion(outputs, targets)
    assert all(torch.isfinite(loss).all() for loss in losses)
    assert sum(loss.item() for loss in losses) > 0
