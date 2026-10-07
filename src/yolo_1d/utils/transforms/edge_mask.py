"""Маскирование неразмеченных краёв записи при загрузке (D5).

Сигнал вне `[first_onset − δ_start, last_offset + δ_end]` заменяется нулями (после Z-score 0 —
среднее значение). Неразмеченные волны на краях записи не попадают в сеть, и лосс не надо менять.

Почему δ случайный на обучении (E1 §3, docs/results/E1_smoke.md): при фиксированном δ положение
границы «нули → сигнал» точно указывает на onset первой и offset последней волны, и сеть
использует это как подсказку. Случайный δ делает положение границы неинформативным.

Диапазоны по умолчанию (D5) выбраны по интервалам между волнами LUDB, чтобы маска не открывала
соседнюю неразмеченную волну:
- начало: перед первой волной (в 93 % — QRS) обычно стоит неразмеченная P; сегмент P→QRS —
  5-й перцентиль 13 отсчётов → δ_start ∈ [0, 12];
- конец: после последней волны (в 93 % — QRS) стоит T; интервал QRS→T — 5-й перцентиль 25
  отсчётов → δ_end ∈ [0, 24].
На валидации и тесте — фиксированные середины диапазонов (6 и 12).
"""
import random
from typing import Optional, Sequence, Tuple

import torch


class EdgeMask:
    """Обнуляет сигнал вне размеченной зоны с запасом δ.

    margin_start, margin_end
        Диапазоны (min, max) запаса в отсчётах (включительно); при min == max запас фиксирован.
    """

    def __init__(self, margin_start: Sequence[int] = (0, 12), margin_end: Sequence[int] = (0, 24)):
        self.margin_start = tuple(int(v) for v in margin_start)
        self.margin_end = tuple(int(v) for v in margin_end)
        for lo, hi in (self.margin_start, self.margin_end):
            if not 0 <= lo <= hi:
                raise ValueError(f"диапазон запаса должен быть 0 <= min <= max, получено ({lo}, {hi})")

    @property
    def random(self) -> bool:
        return self.margin_start[0] != self.margin_start[1] or self.margin_end[0] != self.margin_end[1]

    def zone(self, annotated: Tuple[int, int], length: int) -> Tuple[int, int]:
        """Полуинтервал [start, end) открытого сигнала для разметки (first_onset, last_offset)."""
        first_onset, last_offset = annotated
        start = max(0, first_onset - random.randint(*self.margin_start))
        end = min(length, last_offset + random.randint(*self.margin_end) + 1)
        return start, end

    def __call__(self, signal: torch.Tensor, annotated: Optional[Tuple[int, int]]) -> torch.Tensor:
        """signal (channels, length); annotated — (first_onset, last_offset) или None (без маски)."""
        if annotated is None:
            return signal
        start, end = self.zone(annotated, signal.shape[-1])
        masked = torch.zeros_like(signal)
        masked[..., start:end] = signal[..., start:end]
        return masked

    def __repr__(self) -> str:
        return f"EdgeMask(margin_start={self.margin_start}, margin_end={self.margin_end})"
