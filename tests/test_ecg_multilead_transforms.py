"""Тесты A3: многоканальные аугментации режима MC (ecg_multilead_transforms, склейка в MultiLeadDataset)."""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.settings import AugmentationSettings  # noqa: E402
from yolo_1d.utils.transforms import build_ecg_multilead_training_transforms, build_lead_transforms  # noqa: E402
from yolo_1d.utils.transforms import ecg_multilead_transforms as M  # noqa: E402
from yolo_1d.utils.transforms import ecg_transforms as T  # noqa: E402

L, N = 12, 4000
LEVEL = {0: 1.0, 1: 3.0, 2: 2.0}  # уровень «волны» по классу; изолиния 0.05


def synthetic(zones=True, seed=0):
    """Запись 12 × N: прямоугольные волны P/QRS/T (каждые 800 отсчётов), границы сдвинуты по отведениям
    на ±4 отсчёта; вне зоны отведения — нули (маска краёв, у отведений разная)."""
    rng = np.random.default_rng(seed)
    x = np.full((L, N), 0.05, dtype=np.float32)
    boxes, labels = [], []
    for k in range(L):
        d = int(rng.integers(-4, 5))
        for beat in range(4):
            o = 300 + 800 * beat
            for c, (l, r) in ((0, (o, o + 50)), (1, (o + 100, o + 140)), (2, (o + 250, o + 350))):
                x[k, l + d:r + d] = LEVEL[c]
                boxes.append([(l + r + 2 * d) / 2 / N, (r - l) / N])
                labels.append([c, k])
        if zones:
            x[k, :250 + 3 * k] = 0
            x[k, 3500 - 5 * k:] = 0
    return torch.from_numpy(x), torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def assert_on_waves(x, boxes, labels, tol=0.6):
    """Каждый бокс лежит на своей волне в своём отведении: внутри — уровень класса, сразу снаружи — нет."""
    n = x.shape[-1]
    left, right = T.boxes_to_edges(boxes, n)
    for l, r, (c, k) in zip(left, right, labels.long().tolist()):
        a, b = int(np.ceil(l)) + 2, int(np.floor(r)) - 2
        if b <= a:
            continue
        inside = x[k, a:b].mean().item()
        assert abs(inside - LEVEL[c]) < tol, (c, k, inside)
        for edge in (int(np.floor(l)) - 3, int(np.ceil(r)) + 3):
            if 0 <= edge < n and x[k, edge] != 0:
                assert x[k, edge].item() < LEVEL[c] - tol, (c, k, edge)


def test_wave_clusters_merge_leads():
    x, boxes, labels = synthetic()
    left, right, cls, cluster_of = M.wave_clusters(boxes, labels, N)
    assert len(cls) == 4 * 3 and list(cls[:3]) == [0, 1, 2]
    assert len(np.unique(cluster_of)) == 12


@pytest.mark.parametrize("op", [M.MultiLeadHRVWarp(1.0), T.TimeScale(1.0), T.WindowWarp(1.0),
                                T.FitToLength(5120), T.TimeMask(1.0, max_fraction=0.2)])
def test_time_transforms_keep_boxes_on_waves(op):
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        x, boxes, labels = synthetic(seed=seed)
        y, b, lab = op(x, boxes, labels)
        assert lab.shape[1] == 2 and len(b) == len(lab)
        assert_on_waves(y, b, lab)


@pytest.mark.parametrize("op", [M.CoupledNoise(1.0, "wander"), M.CoupledNoise(1.0, "powerline"),
                                M.LeadBaselineStep(1.0), T.EMGBursts(1.0), T.WhiteNoise(1.0), T.RandomFilter(1.0)])
def test_noise_keeps_each_lead_mask(op):
    random.seed(1)
    np.random.seed(1)
    x, boxes, labels = synthetic()
    y, b, lab = M.KeepZeros(op)(x, boxes, labels)
    assert torch.equal(y[x == 0], x[x == 0])  # нули маски каждого отведения остались
    assert not torch.equal(y, x) and torch.equal(b, boxes) and torch.equal(lab, labels)


def test_coupled_noise_is_one_source():
    """N1/N2: шум во всех отведениях — один источник с коэффициентом отведения (|корреляция| = 1)."""
    for kind in ("wander", "powerline"):
        random.seed(2)
        x = torch.randn(L, N)
        noise = M.CoupledNoise(1.0, kind)(x, None, None)[0] - x
        c = np.corrcoef(noise.numpy())
        assert np.allclose(np.abs(c), 1, atol=1e-4)


@pytest.mark.skipif(not (REPO / "data" / "raw" / "nstdb" / "bw.dat").is_file(), reason="нет NSTDB")
def test_coupled_nstdb_is_one_source():
    random.seed(3)
    x = torch.randn(L, N)
    noise = M.CoupledNoise(1.0, "nstdb")(x, None, None)[0] - x
    assert np.allclose(np.abs(np.corrcoef(noise.numpy())), 1, atol=1e-4)


def test_baseline_step_in_one_to_three_leads():
    for seed in range(10):
        random.seed(seed)
        x = torch.randn(L, N)
        y = M.LeadBaselineStep(1.0)(x, None, None)[0]
        changed = int(((y - x).abs().sum(-1) > 0).sum())
        assert 0 <= changed <= 3


def test_wave_scale_changes_only_own_lead():
    x, boxes, labels = synthetic(zones=False)
    one = labels[:, 1] == 3
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        y, _, _ = M.MultiLeadWaveScale(1.0)(x, boxes[one], labels[one])
        other = torch.ones(L, dtype=torch.bool)
        other[3] = False
        assert torch.equal(y[other], x[other])


def test_amplitude_scale_common_and_per_lead():
    np.random.seed(0)
    x = torch.randn(L, N)
    y = M.MultiLeadAmplitudeScale(1.0, sigma=0.2, sigma_lead=0.1)(x, None, None)[0]
    ratio = (y / x)[:, 0]
    assert (ratio > 0).all() and ratio.std() > 0


# ---------------------------------------------------------------------------------------------- группа L
def test_lead_dropout_ignore_and_keep():
    x, boxes, labels = synthetic()
    valid = torch.ones(L, dtype=torch.bool)
    for mode in ("ignore", "keep"):
        random.seed(4)
        y, b, lab, v = M.LeadDropout(1.0, p_lead=0.3, max_leads=4, labels_mode=mode)(x, boxes, labels, valid)
        dropped = [k for k in range(L) if y[k].abs().sum() == 0]
        assert 1 <= len(dropped) <= 4
        if mode == "ignore":
            assert not v[dropped].any() and v.sum() == L - len(dropped)
            assert not np.isin(lab[:, 1].numpy(), dropped).any()
        else:
            assert v.all() and len(lab) == len(labels)


def test_lead_shuffle_moves_labels_with_signal():
    x, boxes, labels = synthetic()
    x = x * torch.arange(1, L + 1, dtype=torch.float32)[:, None]  # канал узнаётся по масштабу
    valid = torch.ones(L, dtype=torch.bool)
    valid[7] = False
    for seed in range(10):
        random.seed(seed)
        y, b, lab, v = M.LeadShuffle(1.0)(x, boxes, labels, valid)
        src = [int(torch.round(y[j][y[j] != 0].abs().max() / 3.0).item()) - 1 for j in range(L)]  # старый канал
        assert sorted(src) == list(range(L))
        # волна старого отведения i теперь в отведении j, где src[j] = i; ширины/центры не менялись
        old_lead = {j: src[j] for j in range(L)}
        assert torch.equal(b, boxes)
        assert [old_lead[int(k)] for k in lab[:, 1]] == labels[:, 1].long().tolist()
        assert v.tolist() == [valid[src[j]].item() for j in range(L)]


@pytest.mark.parametrize("name,a,b", [("LA-RA", "LA", "RA"), ("LA-LL", "LA", "LL"), ("RA-LL", "RA", "LL")])
def test_electrode_swap_matches_physics(name, a, b):
    """Отведения, посчитанные из электродов после их перестановки, = результат ElectrodeSwap."""
    rng = np.random.default_rng(0)
    el = {e: torch.from_numpy(rng.standard_normal(500)).float() for e in ("RA", "LA", "LL")}
    chest = torch.from_numpy(rng.standard_normal((6, 500))).float()

    def leads(e):
        i, ii = e["LA"] - e["RA"], e["LL"] - e["RA"]
        return torch.cat([torch.stack([i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2]), chest])

    swapped = dict(el)
    swapped[a], swapped[b] = el[b], el[a]
    labels = torch.tensor([[1.0, k] for k in range(L)])
    boxes = torch.tensor([[0.1 + 0.05 * k, 0.01] for k in range(L)])
    y, bx, lab, v = M.ElectrodeSwap.swap(name, leads(el), boxes, labels, torch.ones(L, dtype=torch.bool))
    assert torch.allclose(y, leads(swapped), atol=1e-5)
    # разметка уехала с отведением: волна, бывшая в старом канале src[j], теперь в канале j
    src = M.ELECTRODE_SWAPS[name][0] + list(range(6, L))
    moved = {int(old): int(new) for old, new in zip(labels[:, 1], lab[:, 1])}
    assert all(moved[src[j]] == j for j in range(L))


# ---------------------------------------------------------------------------------------------- датасет и Trainer
needs_data = pytest.mark.skipif(not (REPO / "data" / "processed" / "ludb_mc" / "leads.csv").is_file(),
                                reason="нет data/processed/ludb_mc")
SPLITS = REPO / "data" / "splits"


def full_settings() -> AugmentationSettings:
    config = yaml.safe_load((REPO / "experiments" / "configs" / "final.yaml").read_text(encoding="utf-8"))
    aug = {**config["training"]["augmentations"], "pipeline": "ecg_multi"}
    aug["ecg"] = {**aug["ecg"], "invert_p": 0.0}
    aug["leads"] = {"dropout_p": 0.5, "shuffle_p": 0.1, "electrode_swap_p": 0.2}
    return AugmentationSettings(**aug)


@needs_data
def test_aligned_cuts_are_outside_all_lead_waves():
    from yolo_1d.utils.multilead_dataset import MultiLeadDataset
    ds = MultiLeadDataset(str(SPLITS / "fold_0_mc.yaml"), None, "train")
    for i in range(0, ds.n, 9):
        label = ds.labels[ds.filenames[i]]
        x = ds.load_signal(i)
        cuts = ds._cut_points(ds._merged(label, 5000), 5000, True, ds._active_span(x))
        left = (label[:, 2] - label[:, 3] / 2) * 5000
        right = (label[:, 2] + label[:, 3] / 2) * 5000
        for c in cuts:
            assert not ((left < c) & (c < right)).any()


@needs_data
def test_full_pipeline_samples():
    """Все аугментации MC: формы, разметка в [0, 1], разметка только пригодных отведений."""
    from yolo_1d.utils.multilead_dataset import MultiLeadDataset
    from yolo_1d.utils.transforms import EdgeMask
    settings = full_settings()
    ds = MultiLeadDataset(str(SPLITS / "fold_0_mc.yaml"), build_ecg_multilead_training_transforms(settings), "train",
                          edge_mask=EdgeMask((0, 12), (0, 24)), splice_prob=settings.ecg.mosaic_p,
                          beat_splice_prob=settings.ecg.splice_p, lead_transform=build_lead_transforms(settings))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(40):
        x, cls, box, lead, valid = ds[i % ds.n]
        assert x.shape == (12, 5120) and torch.isfinite(x).all()
        assert len(cls) > 0 and (box[:, 1] > 0).all()
        left, right = box[:, 0] - box[:, 1] / 2, box[:, 0] + box[:, 1] / 2
        assert (left >= -1e-4).all() and (right <= 1 + 1e-4).all()
        assert valid[lead.view(-1).long()].all()


@needs_data
def test_trainer_pipeline_mode_guard():
    from yolo_1d.utils.settings import TrainingSettings
    from yolo_1d.yolo_1d import Trainer
    config = yaml.safe_load((REPO / "experiments" / "configs" / "final.yaml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        training = {**config["training"], "input_channels": 12, "n_leads": 12, "num_workers": 0}
        settings = TrainingSettings(**training, dataset=str(SPLITS / "fold_0_mc.yaml"), results_path=tmp,
                                    project="t", seed=0)
        with pytest.raises(ValueError):  # pipeline 'ecg' (режим S) с датасетом MC
            Trainer(settings).collect_predictions("val")
        training["augmentations"] = {**training["augmentations"], "pipeline": "ecg_multi",
                                     "leads": {"dropout_p": 0.5, "electrode_swap_p": 0.2}}
        settings = TrainingSettings(**training, dataset=str(SPLITS / "fold_0_mc.yaml"), results_path=tmp,
                                    project="t", seed=0)
        trainer = Trainer(settings)
        trainer._init_dataloaders()
        samples, targets = next(iter(trainer._train_loader))
        assert samples.shape[1:] == (12, 5120) and targets["valid"].shape == (samples.shape[0], 12)
        assert trainer._train_dataset.lead_transform is not None
