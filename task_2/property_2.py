import numpy as np
import matplotlib.pyplot as plt

# === Параметры ===
N = 128                     # длина ДПФ
n = np.arange(N)            # дискретное время: n = 0, 1, ..., N-1

f0 = 10                     # частота синуса (целое число → нет утечки!)
n0 = 15                     # сдвиг вправо (задержка)

# Исходный сигнал: x(n) = sin(2π f0 n / N)
x = np.sin(2 * np.pi * f0 * n / N)

# Сдвинутый сигнал: x(n - n0) — циклический сдвиг (корректно для ДПФ)
x_shifted = np.roll(x, shift=n0)  # np.roll делает x[(n - n0) mod N]

# === ДПФ ===
X = np.fft.fft(x)
X_shifted = np.fft.fft(x_shifted)

# Правая часть: X(k) * exp(-j * 2π * n0 * k / N)
k = np.arange(N)
phase_factor = np.exp(-1j * 2 * np.pi * n0 * k / N)
X_right = X * phase_factor

# Проверка численного равенства
print("Спектры совпадают?", np.allclose(X_shifted, X_right, atol=1e-12))

# === Функция для очистки фазы (зануление при малой амплитуде) ===
def clean_phase(fft_vals, threshold=1e-10):
    amp = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    phase[amp < threshold] = 0.0
    return phase

phase_left = clean_phase(X_shifted)
phase_right = clean_phase(X_right)

# === Визуализация ===
plt.figure(figsize=(12, 10))

# 1. Амплитудный спектр
plt.subplot(3, 1, 1)
plt.plot(np.abs(X_shifted), 'b-', label='|DFT{x(n - n₀)}| (левая часть)')
plt.plot(np.abs(X_right),    'r--', label='|X(k)·exp(-j2πn₀k/N)| (правая часть)')
plt.title('Амплитудный спектр: должны совпадать')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

# 2. Фазовый спектр (очищенный)
plt.subplot(3, 1, 2)
plt.plot(phase_left,  'b-', label='Фаза (левая часть)')
plt.plot(phase_right, 'r--', label='Фаза (правая часть)')
plt.title('Фазовый спектр (фаза занулена при малой амплитуде)')
plt.ylabel('Фаза (рад)')
plt.ylim(-np.pi, np.pi)
plt.grid(True)
plt.legend()

# 3. Исходный и сдвинутый сигналы во времени
plt.subplot(3, 1, 3)
plt.plot(n, x, 'g-', label='x(n) = sin(2π f₀ n / N)')
plt.plot(n, x_shifted, 'm--', label=f'x(n - {n0}) — сдвиг вправо')
plt.title('Сигналы во временной области')
plt.xlabel('n')
plt.ylabel('x(n)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()