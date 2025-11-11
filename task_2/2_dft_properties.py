# 2_dft_properties.py
# Проверка свойств Дискретного Преобразования Фурье (ДПФ) через графики

import numpy as np
import matplotlib.pyplot as plt
import os

# Автоматически определяем путь к папке, где лежит скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))

# Параметры сигнала
N = 256  # длина ДПФ (число отсчётов)
n = np.arange(N)  # дискретное время
fs = 100  # частота дискретизации (Гц)
f1 = 5  # частота первого сигнала (Гц)
f2 = 12  # частота второго сигнала (Гц)
t = n / fs  # время в секундах

# Тестовые сигналы
x = np.sin(2 * np.pi * f1 * t)
y = np.sin(2 * np.pi * f2 * t)


# Вспомогательная функция для вычисления ДПФ и центрирования
def compute_dft(signal):
    X = np.fft.fft(signal)
    X_shifted = np.fft.fftshift(X)
    k = np.arange(-N // 2, N // 2)  # индексы частот
    return k, X_shifted


# Функция для построения и сохранения амплитудного и фазового спектров
def plot_spectrum(k, X, filename_suffix):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(k, np.abs(X), 'b')
    plt.title(f'Амплитудный спектр {filename_suffix}')
    plt.xlabel('Индекс частоты k')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k, np.angle(X), 'r')
    plt.title(f'Фазовый спектр {filename_suffix}')
    plt.xlabel('Индекс частоты k')
    plt.grid(True)

    plt.tight_layout()
    filepath = os.path.join(script_dir, f"spectrum_{filename_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png")
    plt.savefig(filepath)
    plt.close()  # закрываем фигуру, чтобы не засорять память


# === 1. Линейность: a*x(n) + b*y(n) ↔ a*X(k) + b*Y(k) ===
print("1. Линейность")
a, b = 1.5, -0.8
left_signal = a * x + b * y
k, X_left = compute_dft(left_signal)

_, X_orig = compute_dft(x)
_, Y_orig = compute_dft(y)
X_right = a * X_orig + b * Y_orig

plot_spectrum(k, X_left, "linear_left")
plot_spectrum(k, X_right, "linear_right")
print("Графики сохранены как spectrum_linear_left.png и spectrum_linear_right.png.\n")

# === 2. Сдвиг во времени: x(n - n0) ↔ X(k)·exp(-2πi·n0·k/N) ===
print("2. Сдвиг во времени")
n0 = 20
x_shifted = np.roll(x, n0)  # циклический сдвиг (требуется для ДПФ)

k, X_left_shift = compute_dft(x_shifted)

_, X_orig = compute_dft(x)
phase_factor = np.exp(-2j * np.pi * n0 * k / N)
X_right_shift = X_orig * phase_factor

plot_spectrum(k, X_left_shift, f"time_shift_left_n0_{n0}")
plot_spectrum(k, X_right_shift, f"time_shift_right_n0_{n0}")
print(f"Графики сдвига сохранены с n0={n0}.\n")

# === 3. Свёртка: (x * y)(n) ↔ X(k)·Y(k) ===
print("3. Свёртка (циклическая)")
conv_xy = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
k, X_left_conv = compute_dft(conv_xy)

_, X_orig = compute_dft(x)
_, Y_orig = compute_dft(y)
X_right_conv = X_orig * Y_orig

plot_spectrum(k, X_left_conv, "convolution_left")
plot_spectrum(k, X_right_conv, "convolution_right")
print("Графики свёртки сохранены.\n")

# === 4. Умножение: x(n)·y(n) ↔ (1/N)·(X * Y)(k) ===
print("4. Умножение во времени")
product_xy = x * y
k, X_left_prod = compute_dft(product_xy)

X_orig = np.fft.fft(x)
Y_orig = np.fft.fft(y)
conv_cyclic = np.array([np.sum(X_orig[m] * Y_orig[(k_idx - m) % N] for m in range(N)) for k_idx in range(N)])
X_right_prod = (1 / N) * np.fft.fftshift(conv_cyclic)

plot_spectrum(k, X_left_prod, "multiplication_left")
plot_spectrum(k, X_right_prod, "multiplication_right")
print("Графики умножения сохранены.\n")

print("Проверка завершена. Все графики сохранены в папку со скриптом.")