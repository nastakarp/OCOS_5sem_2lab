# 4_stft_spectrograms.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft, windows
import os
import re

# Определяем папку скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))

# Параметры
fs = 100  # частота дискретизации
T = 3.0  # длительность, сек
t = np.arange(0, T, 1 / fs)
N = len(t)

f1, f2, f3 = 1.0, 2.0, 3.0

# -------------------------------------------------
# 1. Генерация сигналов y1, y2, y3
# -------------------------------------------------
y1 = np.zeros_like(t)
y1[(t >= 0) & (t < 1)] = np.sin(2 * np.pi * f1 * t[(t >= 0) & (t < 1)])
y1[(t >= 1) & (t < 2)] = np.sin(2 * np.pi * f2 * t[(t >= 1) & (t < 2)])
y1[(t >= 2) & (t < 3)] = np.sin(2 * np.pi * f3 * t[(t >= 2) & (t < 3)])

y2 = (1 / 3) * (np.sin(2 * np.pi * f1 * t) +
                np.sin(2 * np.pi * f2 * t) +
                np.sin(2 * np.pi * f3 * t))

phase_y3 = 2 * np.pi * (0.5 * t ** 2 + 0.5 * t)
y3 = np.sin(phase_y3)

# Сигналы и их человекочитаемые названия (для графиков)
signals = [y1, y2, y3]
signal_titles = [
    'y1 (последовательные тона)',
    'y2 (сумма трёх тонов)',
    'y3 (ЧМ-сигнал: f = t + 0.5)'
]

# Безопасные латинские имена для файлов
signal_filenames = ['y1_sequential', 'y2_sum_of_tones', 'y3_fm_signal']

# -------------------------------------------------
# 2. Амплитудные спектры (FFT)
# -------------------------------------------------
plt.figure(figsize=(15, 4))
for i, (sig, title) in enumerate(zip(signals, signal_titles)):
    X = fft(sig)
    freqs = fftfreq(N, d=1 / fs)
    amp = np.abs(X) / N

    plt.subplot(1, 3, i + 1)
    plt.plot(freqs[:N // 2], amp[:N // 2] * 2)
    plt.title(f'Амплитудный спектр: {title}')
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.xlim(0, 5)

plt.tight_layout()
fft_path = os.path.join(script_dir, 'fft_spectra.png')
plt.savefig(fft_path, dpi=150)
plt.close()
print("FFT спектры сохранены: fft_spectra.png")

# -------------------------------------------------
# 3. Спектрограммы (STFT) с разными окнами
# -------------------------------------------------
window_percentages = [1, 10, 30]
nfft = 512

for idx, (sig, title, fname) in enumerate(zip(signals, signal_titles, signal_filenames)):
    plt.figure(figsize=(16, 10))
    plt.suptitle(f'Спектрограммы сигнала: {title}', fontsize=14)

    for j, pct in enumerate(window_percentages):
        nperseg = int(N * pct / 100)
        nperseg = max(4, min(nperseg, 512))

        f, t_stft, Zxx = stft(
            sig, fs=fs,
            window='hamming',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=nfft,
            boundary=None,
            padded=False
        )

        magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

        plt.subplot(3, 1, j + 1)
        im = plt.pcolormesh(t_stft, f, magnitude_db, shading='gouraud', cmap='viridis')
        plt.title(f'Окно Хэмминга: {pct}% от длины сигнала (nperseg = {nperseg})')
        plt.ylabel('Частота, Гц')
        plt.ylim(0, 5)
        plt.colorbar(im, label='Амплитуда (дБ)')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Время, с')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Сохраняем с чистым латинским именем
    spectrogram_path = os.path.join(script_dir, f'spectrogram_{idx + 1}_{fname}.png')
    plt.savefig(spectrogram_path, dpi=150)
    plt.close()
    print(f"Спектрограммы сохранены: spectrogram_{idx + 1}_{fname}.png")

# -------------------------------------------------
# 4. Выводы
# -------------------------------------------------
print("\n=== Выводы по спектрограммам ===")
print("• При малом окне (1%): высокое разрешение по времени, низкое по частоте.")
print("  → Видны моменты включения/выключения тонов (в y1), но частоты размыты.")
print("• При большом окне (30%): высокое разрешение по частоте, низкое по времени.")
print("  → Чёткие частоты (1,2,3 Гц), но не видно, когда они включаются (в y1).")
print(
    "• Для ЧМ-сигнала (y3): оптимальный баланс — среднее окно (10%) лучше всего отображает изменение частоты во времени.")