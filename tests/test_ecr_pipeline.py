"""Тесты сборки конвейеров ECRv1 (build_transforms, задача D6): порядок шагов, RandomApply, выключение групп,
кэш val/test, выбор конвейера по датасету, прогон через Dataset + DataLoader."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils import ecr  # noqa: E402
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms.transforms import RandomApply  # noqa: E402
from yolo_1d.utils.transforms import (EddyPreprocessing, PreprocessedCache,  # noqa: E402
                                      ReferenceStandardize, build_ecr_training_transforms,
                                      build_transforms_for_dataset, build_validation_transform)

REG, INP = 6000, 4096
OFF = dict(regular_length=REG, input_length=INP, random_resize=0, random_crop=False, phase_spectrum_noise=0,
           random_jitter=0, random_invert=0, random_reverse=0, channel_scaling=0, time_masking=0,
           channel_masking=0, mosaic=0, mix_up=0, sine_noise=0, square_noises=0)
FULL = dict(regular_length=REG, input_length=INP, random_resize=0.1, random_crop=True, phase_spectrum_noise=0.5,
            random_jitter=0.6, random_invert=0.5, random_reverse=0.5, channel_scaling=0.2, time_masking=0.2,
            channel_masking=0.2, mosaic=0.5, mix_up=0.5, sine_noise=0.5, square_noises=0.1)

# таблица шагов 1–13 (TASKS.md) при всех включённых группах
EXPECTED_ORDER = ["ReferenceStandardize", "RandomResize", "RandomCrop", "PhaseSpectrumNoise", "RandomJitter",
                  "RandomInvert", "RandomReverse", "SineNoise", "SquareNoises", "ICASuppression", "Destandardize",
                  "FourierAlign", "TimeMasking", "ThresholdNormalize", "ChannelScaling", "ChannelMasking",
                  "RandomPadding"]
NOT_WRAPPED = {"ReferenceStandardize", "RandomResize", "RandomCrop", "ICASuppression", "Destandardize",
               "FourierAlign", "ThresholdNormalize", "RandomPadding"}


def synthetic(length=REG, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 40, size=(15, length))
    for c in rng.integers(300, length - 300, size=3):
        x[:, c - 60:c + 60] += rng.normal(0, 400, size=(15, 1)) * np.hanning(120)
    return np.round(x).astype(np.int16)


def names(compose):
    out = []
    for op in compose.transforms:
        inner = op.augmentation if isinstance(op, RandomApply) else op
        out.append((type(inner).__name__, op.p if isinstance(op, RandomApply) else None))
    return out


def run(t, x, box=None, cls=None):
    box = torch.tensor([[0.5, 0.1]]) if box is None else box
    cls = torch.tensor([[1.0]]) if cls is None else cls
    return t(torch.from_numpy(x.astype(np.float32)), box, cls)


def test_order_matches_table_and_random_apply():
    s = AugmentationSettings(**FULL)
    got = names(build_ecr_training_transforms(s, EddyPreprocessing()))
    assert [n for n, _ in got] == EXPECTED_ORDER
    probs = dict(got)
    for n in NOT_WRAPPED:
        assert probs[n] is None, n
    assert probs["SineNoise"] == pytest.approx(0.5 * 0.5)
    assert probs["SquareNoises"] == pytest.approx(0.5 * 0.1)
    for n in ("PhaseSpectrumNoise", "RandomJitter", "RandomInvert", "RandomReverse", "TimeMasking",
              "ChannelScaling", "ChannelMasking"):
        assert probs[n] == pytest.approx(0.5), n


def test_groups_switch_off():
    got = [n for n, _ in names(build_ecr_training_transforms(AugmentationSettings(**OFF), EddyPreprocessing()))]
    assert got == ["ReferenceStandardize", "ICASuppression", "Destandardize", "FourierAlign", "ThresholdNormalize",
                   "Padding", "Resize"]


def test_mosaic_requires_random_crop():
    with pytest.raises(ValueError):
        build_ecr_training_transforms(AugmentationSettings(**{**OFF, "mosaic": 0.5}), EddyPreprocessing())


def test_all_off_train_equals_validation():
    s = AugmentationSettings(**OFF)
    x = synthetic(seed=1)
    tr, b1, c1 = run(build_ecr_training_transforms(s, EddyPreprocessing()), x)
    va, b2, c2 = run(build_validation_transform(s, EddyPreprocessing()), x)
    assert tr.shape == (30, INP)
    torch.testing.assert_close(tr, va, rtol=0, atol=0)
    torch.testing.assert_close(b1, b2)


@pytest.mark.parametrize("length", [REG - 1500, REG, 2 * REG])
def test_full_pipeline_output(length):
    np.random.seed(0)
    torch.manual_seed(0)
    t = build_ecr_training_transforms(AugmentationSettings(**FULL), EddyPreprocessing())
    for seed in range(3):
        y, box, cls = run(t, synthetic(length, seed))
        assert y.shape == (30, INP) and y.dtype == torch.float32 and torch.isfinite(y).all()
        assert len(box) == len(cls)


def test_validation_cache(tmp_path):
    s = AugmentationSettings(**OFF)
    x = synthetic(seed=4)
    plain, _, _ = run(build_validation_transform(s, EddyPreprocessing()), x)
    t = build_validation_transform(s, EddyPreprocessing(cache_dir=str(tmp_path)))
    assert isinstance(t.transforms[0], PreprocessedCache)
    miss, _, _ = run(t, x)
    assert len(list((tmp_path / "preprocessed").rglob("*.npy"))) == 1
    hit, _, _ = run(t, x)
    torch.testing.assert_close(miss, hit, rtol=0, atol=0)          # результат не зависит от состояния кэша
    # кэш хранит float16 до Padding/Resize: сравнение с некэшированным — с точностью float16
    torch.testing.assert_close(miss, plain, rtol=1e-3, atol=1e-2)


def make_ecr(root: Path, n=4):
    for split in ecr.SPLITS:
        (root / split / "data").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            np.save(root / split / "data" / f"p1_left_{i:04d}.npy", synthetic(REG - 500 * i, seed=i))
            (root / split / "labels" / f"p1_left_{i:04d}.txt").write_text(f"{1 + i % 5} 1000 1400\n12 3000 3150\n")
    return ecr.write_manifest(root, root / "cfg" / "ecr.yaml")


def test_transforms_for_dataset_selection(tmp_path):
    m = make_ecr(tmp_path)
    tr, te = build_transforms_for_dataset(AugmentationSettings(**FULL), str(m))
    assert isinstance(tr.transforms[0], ReferenceStandardize) and tr.transforms[0].cache is not None
    assert isinstance(te.transforms[0], PreprocessedCache)
    assert tr.transforms[0].context is not te.transforms[0].ops[0].context      # свой контекст у train и val
    with pytest.raises(ValueError):
        build_transforms_for_dataset(AugmentationSettings(**{**FULL, "pipeline": "ecg"}), str(m))
    tr, te = build_transforms_for_dataset(AugmentationSettings(), None)        # прежний конвейер
    assert not any(isinstance(op, ReferenceStandardize) for op in tr.transforms)


def test_dataset_dataloader_workers(tmp_path):
    m = make_ecr(tmp_path)
    s = AugmentationSettings(**FULL)
    tr, te = build_transforms_for_dataset(s, str(m))
    train = Dataset(str(m), tr, "train", mosaic_prob=s.mosaic, mix_up_prob=s.mix_up)
    val = Dataset(str(m), te, "val")
    for ds in (train, val):
        loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=Dataset.collate_fn)
        for x, t in loader:
            assert x.shape == (2, 30, INP) and torch.isfinite(x).all()
            assert ((t["box"] >= 0) & (t["box"] <= 1)).all() and len(t["cls"]) == len(t["box"]) == len(t["idx"])
