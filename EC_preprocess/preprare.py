
import numpy as np
from scipy.fft import irfft, rfft

from .signal import Signal
from .kuzmin_levels import Get_Signal_Thresholds, New_Channel_Substitution

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

def prepare_signal(data:np.ndarray) -> Signal:
    
    data = ICA_filtration(data, median=True, EMA=True, method='hyvarinen')
    # Фурье выравнивание
    data = data - fftine_fast(data, convert_freq(0.03, data.shape[1]))
    # вычисление пороговых уровней
    _Threshold1, _Threshold2, _CorrectedThreshold1, _CorrectedThreshold2= [], [], [], []
    Get_Signal_Thresholds(
            data.T,
            _Threshold1,
            _Threshold2,
            _CorrectedThreshold1,
            _CorrectedThreshold2,
        )
    # Замена битых каналов
    New_Channel_Substitution(data.T,_Threshold1,_Threshold2,_CorrectedThreshold1,_CorrectedThreshold2)
    # Сохранение нужного порогового уровня
    signal = Signal(data)
    signal.sigma_plus = np.array([min(abs(item1.thplus)/3,abs(item2.thplus)/3) for item1,item2 in zip(_CorrectedThreshold2,_Threshold2)])
    signal.sigma_mins = np.array([min(abs(item1.thmins)/3,abs(item2.thmins)/3) for item1,item2 in zip(_CorrectedThreshold2,_Threshold2)])
    signal.sigma = np.array(min(abs(item1.thplus), abs(item1.thmins)) for item1 in _Threshold1) / 3
    return signal