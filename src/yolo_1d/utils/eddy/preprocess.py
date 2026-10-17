"""Предобработка вихретоковой дефектограммы (15 каналов) для сети YOLO-1d.

Перенесено из `EC_preprocess/preprare.py` без изменения логики (`fftine_fast`, `convert_freq`, `prepare_signal`):

1. подавление импульсных и прямоугольных помех в базисе ICA — `ICA_filtration(median=True, EMA=True,
   method='hyvarinen')` из `transforms/square_noises.py`, параметры как в аугментации `SquareNoises`;
2. Фурье-выравнивание — вычитание низкочастотных гармоник ниже `convert_freq(0.03, L)`
   (спектральное вычитание периодических помех, Быстров, Гладков, Кузьмин, МАИС 2024);
3. пороговые уровни шума рельса и признак битого канала (`Get_Signal_Thresholds`, Кузьмин и др.);
4. замена битых каналов ближайшим исправным (`New_Channel_Substitution`);
5. нормирующий уровень σ = min(|T⁺|, |T⁻|) / 3 по порогам `Threshold1` (решение пользователя 2026-10-16).

Отличия от исходника (согласованы в TASKS.md, D1):
- в строке `signal.sigma = np.array(<генератор>) / 3` исправлена опечатка (нужен список, иначе TypeError);
- `prepare_signal` дополнительно сохраняет карту замен `substitute`: её нужно получить до
  `New_Channel_Substitution`, т.к. та копирует признак `badchl` исправного соседа в битый канал;
- `preprocess` применяет ту же замену к сырой ветке (без подавления помех) — решение пользователя;
- σ = 0 (все каналы битые, заменить нечем) заменяется на 1 и отмечается в результате.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from scipy.fft import irfft, rfft

from yolo_1d.utils.transforms.square_noises import ICA_filtration
from .kuzmin_levels import Get_Signal_Thresholds, New_Channel_Substitution, find_closest
from .signal import Signal

# число каналов дефектограммы (алгоритм порогов рассчитан ровно на 15)
N_CHANNELS = 15


def fftine_fast(data, to=None):
    channels, length = data.shape
    to = 0 if to is None else to

    # Используем векторные операции вместо цикла
    ff = rfft(data, axis=1)  # Прямое преобразование для всех каналов сразу
    ff[:, to:] = 0  # Обрезание фурье коэффициентов для всех каналов
    rez = np.round(irfft(ff, n=length, axis=1)).astype(np.int32)  # Обратное преобразование
    return rez


def convert_freq(freq: float, x: int) -> int:
    return np.ceil(freq / 2 / np.pi * x).astype("int")


def prepare_signal(data: np.ndarray) -> Signal:
    """Подавление помех, пороговые уровни и замена битых каналов (логика `EC_preprocess/preprare.py`).

    data: дефектограмма (15, L). Возвращает `Signal` с полями `data` (очищенный сигнал), `sigma_plus`,
    `sigma_mins`, `sigma` и `substitute` (индекс канала-замены для каждого канала; −1 — заменить нечем).
    """
    data = ICA_filtration(data, median=True, EMA=True, method='hyvarinen')
    # Фурье выравнивание
    data = data - fftine_fast(data, convert_freq(0.03, data.shape[1]))
    # вычисление пороговых уровней
    _Threshold1, _Threshold2, _CorrectedThreshold1, _CorrectedThreshold2 = [], [], [], []
    Get_Signal_Thresholds(
            data.T,
            _Threshold1,
            _Threshold2,
            _CorrectedThreshold1,
            _CorrectedThreshold2,
        )
    # карта замен — до New_Channel_Substitution (она перезаписывает badchl битых каналов)
    substitute = find_closest(_Threshold1)
    # Замена битых каналов
    New_Channel_Substitution(data.T, _Threshold1, _Threshold2, _CorrectedThreshold1, _CorrectedThreshold2)
    # Сохранение нужного порогового уровня
    signal = Signal(data)
    signal.sigma = np.array([min(abs(item1.thplus), abs(item1.thmins)) for item1 in _CorrectedThreshold1]) / 3
    signal.substitute = substitute
    return signal


def substitute_channels(data: np.ndarray, substitute: List[int]) -> np.ndarray:
    """Копия `data` (15, L), где битые каналы заменены каналами из карты `substitute` (−1 — без замены)."""
    out = data.copy()
    for line, src in enumerate(substitute):
        if src != line and src != -1:
            out[line] = data[src]
    return out


@dataclass
class PreprocessResult:
    """Результат предобработки одного фрагмента.

    raw:        (15, L) float32 — сигнал без подавления помех, битые каналы заменены, делён на σ;
    filtered:   (15, L) float32 — сигнал с подавлением помех (ICA + Фурье), битые каналы заменены, делён на σ;
    sigma:      (15,) — нормирующий уровень по каналу (после защиты от нуля);
    substitute: индекс канала-замены для каждого канала (совпадает с номером — канал исправен; −1 — заменить нечем);
    bad_channels:   каналы, признанные битыми;
    sigma_fallback: каналы, у которых σ = 0 заменено на 1.
    """
    raw: np.ndarray
    filtered: np.ndarray
    sigma: np.ndarray
    substitute: List[int]
    bad_channels: List[int] = field(default_factory=list)
    sigma_fallback: List[int] = field(default_factory=list)

    def to_tensor(self) -> torch.Tensor:
        """Тензор входа сети (30, L) float16: `[0:15]` — без подавления помех, `[15:30]` — с подавлением.
        Такой же формат ожидают `Dataset` и аугментация `SquareNoises`."""
        return torch.from_numpy(np.concatenate([self.raw, self.filtered], axis=0).astype(np.float16))


def preprocess(raw: np.ndarray) -> PreprocessResult:
    """Полная предобработка фрагмента дефектограммы.

    raw: (15, L) — исходные отсчёты (int16 из ECRv1 или любой числовой тип).
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[0] != N_CHANNELS:
        raise ValueError(f"expected array of shape ({N_CHANNELS}, L), got {raw.shape}")
    raw = raw.astype(np.float64)

    signal = prepare_signal(raw)
    sigma = signal.sigma.astype(np.float64)
    fallback = [int(i) for i in np.flatnonzero(sigma <= 0)]
    sigma[fallback] = 1.0

    raw_sub = substitute_channels(raw, signal.substitute)
    bad = [i for i, s in enumerate(signal.substitute) if s != i]
    return PreprocessResult(
        raw=(raw_sub / sigma[:, None]).astype(np.float32),
        filtered=(signal.data / sigma[:, None]).astype(np.float32),
        sigma=sigma,
        substitute=list(signal.substitute),
        bad_channels=bad,
        sigma_fallback=fallback,
    )
