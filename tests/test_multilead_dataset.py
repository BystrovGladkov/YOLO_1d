"""Тесты режима MC (M6): данные data/prepare_ludb_mc.py, разбиения data/make_splits_mc.py, MultiLeadDataset.

Главное свойство: вход и разметка каждого отведения в MC совпадают с режимом S (те же файлы, та же маска),
поэтому метрики S и MC сравнимы напрямую.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.utils.dataset import Dataset  # noqa: E402
from yolo_1d.utils.ludb import LEADS  # noqa: E402
from yolo_1d.utils.multilead_dataset import MultiLeadDataset, dataset_class  # noqa: E402
from yolo_1d.utils.transforms import EdgeMask, Padding  # noqa: E402

SPLITS = REPO / "data" / "splits"
S_DIR = REPO / "data" / "processed" / "ludb"
MC_DIR = REPO / "data" / "processed" / "ludb_mc"
needs_data = pytest.mark.skipif(not (MC_DIR / "leads.csv").is_file() or not (S_DIR / "manifest.csv").is_file(),
                                reason="нет data/processed/ludb_mc (python data/prepare_ludb_mc.py)")


class Identity:
    @staticmethod
    def forward(x, box, meta):
        return x, box, meta


def mc(split="test", fold=0, mask=None, transform=Identity()):
    return MultiLeadDataset(str(SPLITS / f"fold_{fold}_mc.yaml"), transform, split, edge_mask=mask)


def s_dataset(split="test", fold=0, mask=None):
    return Dataset(str(SPLITS / f"fold_{fold}.yaml"), Identity(), split, edge_mask=mask)


@needs_data
def test_dispatch_by_mode():
    assert dataset_class(SPLITS / "fold_0_mc.yaml") is MultiLeadDataset
    assert dataset_class(SPLITS / "fold_0.yaml") is Dataset


@needs_data
@pytest.mark.parametrize("fold", range(5))
def test_splits_match_single_lead(fold):
    """Записи сплитов MC = записи отведений сплитов S; отведения S = пригодные отведения записей MC."""
    for split in ("train", "val", "test"):
        s_ids = (SPLITS / f"fold_{fold}" / f"{split}.txt").read_text(encoding="utf-8").split()
        m_ids = (SPLITS / f"fold_{fold}" / f"{split}_mc.txt").read_text(encoding="utf-8").split()
        assert sorted({i.rsplit("_", 1)[0] for i in s_ids}, key=int) == sorted(m_ids, key=int)
        ds = mc(split, fold)
        valid_ids = {f"{ds.sample_id(i)}_{LEADS[k]}" for i in range(ds.n) for k in range(12) if ds.valid(i)[k]}
        assert valid_ids == set(s_ids)


@needs_data
def test_signals_and_labels_equal_single_lead():
    """Канал k записи MC = файл отведения S; разметка отведения k = разметка S (без маски и с маской eval)."""
    eval_mask = EdgeMask((6, 6), (12, 12))
    for mask in (None, eval_mask):
        ds, s = mc(mask=mask), s_dataset(mask=mask)
        s_index = {Path(f).stem: i for i, f in enumerate(s.filenames)}
        for i in range(0, ds.n, 7):
            x = ds.load_signal(i)
            assert x.shape == (12, 5000)
            for k, lead in enumerate(LEADS):
                sid = f"{ds.sample_id(i)}_{lead}"
                if sid not in s_index:
                    continue
                j = s_index[sid]
                assert torch.equal(x[k:k + 1], s.load_signal(j)), sid
                ref = s.labels[s.filenames[j]]
                got = ds.lead_labels(i, k)
                assert np.allclose(np.sort(got, axis=0), np.sort(ref, axis=0), atol=1e-6), sid


@needs_data
def test_unusable_leads_ignored():
    """Непригодные отведения: сигнал во входе, разметки нет, valid = False."""
    with open(MC_DIR / "leads.csv", encoding="utf-8") as f:
        bad = [r for r in csv.DictReader(f) if r["usable"] == "0"]
    assert len(bad) == 24
    records = {r["record"] for r in bad}
    found = 0
    for fold in range(5):
        ds = mc("test", fold)
        for i in range(ds.n):
            if ds.sample_id(i) not in records:
                continue
            valid = ds.valid(i)
            label = ds.labels[ds.filenames[i]]
            x = ds.load_signal(i)
            for r in bad:
                if r["record"] == ds.sample_id(i):
                    k = int(r["lead_index"])
                    assert not valid[k] and not (label[:, 1] == k).any() and x[k].abs().sum() > 0
                    found += 1
    assert found == 24


@needs_data
def test_edge_mask_per_lead_independent():
    """Фиксированная маска: каждое отведение нулевое вне своей зоны ± запас; случайная — δ независим по отведениям."""
    ds = mc(mask=EdgeMask((6, 6), (12, 12)))
    raw = mc()
    i = 0
    x, x_raw = ds.load_signal(i), raw.load_signal(i)
    zones = ds.zones[ds.sample_id(i)]["zones"]
    for k, (start, end) in enumerate(zones):
        lo, hi = max(0, start - 6), min(5000, end + 12 + 1)
        assert torch.equal(x[k, lo:hi], x_raw[k, lo:hi])
        assert x[k, :lo].abs().sum() == 0 and x[k, hi:].abs().sum() == 0
    # случайный запас: начала открытого сигнала относительно зоны различаются между отведениями
    rnd = mc(mask=EdgeMask((0, 12), (0, 24)))
    torch.manual_seed(0)
    offsets = set()
    for _ in range(5):
        x = rnd.load_signal(i)
        for k, (start, _) in enumerate(zones):
            first = int(torch.nonzero(x[k]).flatten()[0])
            offsets.add(start - first)
    assert len(offsets) > 3


@needs_data
def test_getitem_and_collate():
    ds = mc(mask=EdgeMask((6, 6), (12, 12)), transform=Padding(5120))
    batch = [ds[i] for i in range(3)]
    x, cls, box, lead, valid = batch[0]
    assert x.shape == (12, 5120) and valid.shape == (12,) and valid.dtype == torch.bool
    assert cls.shape[1] == 1 and lead.shape == cls.shape and box.shape == (len(cls), 2)
    assert set(lead.view(-1).long().tolist()) <= set(range(12))
    # паддинг масштабирует боксы на 5000/5120, как в режиме S
    raw = ds.labels[ds.filenames[0]]
    assert torch.allclose(box[:, 0].sort().values, torch.from_numpy(raw[:, 2] * 5000 / 5120).sort().values, atol=1e-6)
    samples, targets = MultiLeadDataset.collate_fn(batch)
    assert samples.shape == (3, 12, 5120) and targets["valid"].shape == (3, 12)
    n = sum(len(b[1]) for b in batch)
    assert targets["cls"].shape == (n, 1) and targets["box"].shape == (n, 2) and targets["lead"].shape == (n,)
    assert targets["idx"].tolist() == sum(([float(i)] * len(b[1]) for i, b in enumerate(batch)), [])


def test_mosaic_and_mixup_not_supported(tmp_path):
    for kw in ({"mosaic_prob": 0.2}, {"mix_up_prob": 0.1}):
        with pytest.raises(NotImplementedError):
            MultiLeadDataset(str(tmp_path / "x.yaml"), Identity(), "train", **kw)
