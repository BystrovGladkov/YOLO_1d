"""Датасет режима MC (M6): пример — вся запись LUDB, 12 отведений, разметка каждого отведения.

Формат (data/prepare_ludb_mc.py, data/make_splits_mc.py, конфиг `data/splits/fold_<k>_mc.yaml`, `mode: multi_lead`):
    <split>_data/<запись>.pt     torch.float32 (12, 5000), Z-score по отведению (как в режиме S)
    <split>_labels/<запись>.txt  строки "class lead center width" (lead — индекс 0..11)
    zones (leads.csv)            запись × отведение: annot_start, annot_end, usable

Решения R5 (docs/research/R5_multilead.md §2.1–2.2):
- маска неразмеченных краёв (`EdgeMask`) применяется к **каждому отведению по его зоне**; запас δ
  тянется **независимо для каждого отведения** (на train он случайный, на eval фиксированный);
- непригодные отведения (usable = 0) остаются во входе, но помечаются в `valid` — их нет в лоссе и метриках.

Выход `__getitem__`: (signal (12, L), cls (n, 1), box (n, 2), lead (n, 1), valid (12,) bool).
`collate_fn` добавляет к целям поля `lead` (n,) и `valid` (B, 12). Склейка/мозаика/mix-up для MC — в A3.
"""
import csv
import os
from typing import List, Optional, Tuple

import numpy
import torch
import yaml

from yolo_1d.utils.dataset import Dataset

N_LEADS = 12


class MultiLeadDataset(Dataset):
    """Записи LUDB 12 × 5000 с поотведенческой разметкой. Интерфейс как у `Dataset` (режим S)."""

    n_leads = N_LEADS

    def __init__(self, filename, transform, type, edge_mask=None, **augmentations):
        # вероятности склейки/мозаики/mix-up (параметры вроде mix_up_alpha не важны, пока mix_up_prob = 0)
        unsupported = {k: v for k, v in augmentations.items() if k.endswith("_prob") and v}
        if unsupported:
            raise NotImplementedError(f"в режиме MC пока не поддерживаются: {sorted(unsupported)} (задача A3)")
        super().__init__(filename, transform, type, edge_mask=edge_mask)

    # ------------------------------------------------------------------ чтение
    @staticmethod
    def _read_zones(path) -> Optional[dict]:
        """leads.csv → {запись: {"zones": [(start, end) | None] * 12, "valid": [bool] * 12}}."""
        if path is None:
            return None
        out = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rec = out.setdefault(row["record"], {"zones": [None] * N_LEADS, "valid": [False] * N_LEADS})
                k = int(row["lead_index"])
                rec["zones"][k] = (int(row["annot_start"]), int(row["annot_end"]))
                rec["valid"][k] = row["usable"] == "1"
        return out

    @staticmethod
    def _read_label_file(label_file):
        """Разметка записи (n, 4) [class, lead, center, width] или None, если она некорректна."""
        if not os.path.isfile(label_file):
            return numpy.zeros((0, 4), dtype=numpy.float32)
        with open(label_file) as f:
            rows = [x.split() for x in f.read().strip().splitlines() if x.strip()]
        if not rows:
            return numpy.zeros((0, 4), dtype=numpy.float32)
        label = numpy.array(rows, dtype=numpy.float32)
        if label.shape[1] != 4 or (label < 0).any() or (label[:, 2:] > 1).any() or (label[:, 1] >= N_LEADS).any():
            return None
        return numpy.unique(label, axis=0)

    def sample_id(self, i: int) -> str:
        return os.path.basename(self.filenames[i]).rsplit(".", 1)[0]

    def valid(self, i: int) -> torch.Tensor:
        """(12,) bool: отведения записи, которые входят в лосс и метрики."""
        if self.zones is None:
            return torch.ones(N_LEADS, dtype=torch.bool)
        return torch.tensor(self.zones[self.sample_id(i)]["valid"], dtype=torch.bool)

    def load_signal(self, i) -> torch.Tensor:
        """Сигнал записи (12, 5000); каждое отведение маскируется по своей зоне с независимым δ."""
        with open(self.filenames[i], "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=False).float()
        if self.edge_mask is not None and self.zones is not None:
            zones = self.zones[self.sample_id(i)]["zones"]
            data = torch.cat([self.edge_mask(data[k:k + 1], zones[k]) for k in range(data.shape[0])], dim=0)
        return data

    # ------------------------------------------------------------------ пример и батч
    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        index = self.indices[index]
        data = self.load_signal(index)
        label = self.labels[self.filenames[index]]
        nl = len(label)
        box = torch.from_numpy(label[:, 2:4].copy()) if nl else torch.zeros((0, 2))
        # трансформации получают метки (n, 2) [class, lead]: отбор/перестановка волн сохраняет отведение
        meta = torch.from_numpy(label[:, 0:2].copy()) if nl else torch.zeros((0, 2))
        data, box, meta = self.transform.forward(data, box, meta)
        return data, meta[:, 0:1], box, meta[:, 1:2], self.valid(index)

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, lead, valid = zip(*batch)
        idx = torch.cat([torch.full((len(c),), i, dtype=torch.float32) for i, c in enumerate(cls)])
        targets = {"cls": torch.cat(cls, dim=0),
                   "box": torch.cat(box, dim=0),
                   "idx": idx,
                   "lead": torch.cat(lead, dim=0).view(-1),
                   "valid": torch.stack(valid, dim=0)}
        return torch.stack(samples, dim=0), targets

    # ------------------------------------------------------------------ для метрик
    def lead_ids(self) -> List[str]:
        """ID пригодных отведений `<запись>_<отведение>` в порядке строк выхода модели при последовательном
        проходе датасета (запись за записью, отведения 0..11, непригодные пропущены) — как ID режима S."""
        from yolo_1d.utils.ludb import LEADS
        return [f"{self.sample_id(i)}_{LEADS[k]}" for i in range(self.n) for k in range(N_LEADS) if self.valid(i)[k]]

    def lead_labels(self, i: int, k: int) -> numpy.ndarray:
        """Разметка отведения k записи i в формате режима S (m, 3) [class, center, width]."""
        label = self.labels[self.filenames[i]]
        return label[label[:, 1] == k][:, [0, 2, 3]]


def dataset_class(filename) -> type:
    """Класс датасета по ключу `mode` yaml-конфига: `multi_lead` → MultiLeadDataset, иначе Dataset."""
    with open(filename, encoding="utf-8") as f:
        mode = (yaml.safe_load(f) or {}).get("mode", "single_lead")
    return MultiLeadDataset if mode == "multi_lead" else Dataset
