import numpy as np
import matplotlib.pyplot as plt

# === Параметры ===
N = 128
n = np.arange(N)

f1 = 8
f2 = 5

x = np.sin(2 * np.pi * f1 * n / N)
y = np.sin(2 * np.pi * f2 * n / N)

# ДПФ
X = np.fft.fft(x)
Y = np.fft.fft(y)

# Правая часть: (1/N) * циклическая свёртка спектров X и Y
def cyclic_convolution_freq(X, Y):
    N = len(X)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for m in range(N):
            result[k] += X[m] * Y[(k - m) % N]
    return result

conv_XY = cyclic_convolution_freq(X, Y)
right_side = (1.0 / N) * conv_XY

# Левая часть: DFT{x(n) * y(n)}
product_time = x * y
left_side = np.fft.fft(product_time)

# Проверка
print("DFT{x·y} == (1/N)·(X * Y) ?", np.allclose(left_side, right_side, atol=1e-12))

# Альтернатива: использовать ifft + fft trick (намного быстрее!)
# Циклическая свёртка в частоте: ifft(fft(X) * fft(Y)) — не нужно, т.к. X и Y уже в частоте.
# Но можно сделать: conv = np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y)) — и это будет цикл. свёртка
# Однако это менее наглядно. Оставим ручную для чистоты.

# === Функция для очистки фазы ===
def clean_phase(fft_vals, threshold=1e-10):
    amp = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    phase[amp < threshold] = 0.0
    return phase

phase_left = clean_phase(left_side)
phase_right = clean_phase(right_side)

# === Визуализация ===
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(np.abs(left_side), 'b-', label='|DFT{x·y}|')
plt.plot(np.abs(right_side), 'r--', label='|(1/N)(X*Y)|')
plt.title('Свойство умножения: амплитудный спектр')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(phase_left, 'b-', label='Фаза (DFT{x·y})')
plt.plot(phase_right, 'r--', label='Фаза ((1/N)(X*Y))')
plt.title('Свойство умножения: фазовый спектр')
plt.xlabel('k')
plt.ylabel('Фаза (рад)')
plt.ylim(-np.pi, np.pi)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# === Теоретические частоты произведения синусов ===
# sin(A) * sin(B) = 0.5[cos(A-B) - cos(A+B)]
print(f"\\nОжидаемые пики от x(n)y(n) = sin({f1})·sin({f2}) на частотах:")
print(f"  |f1 - f2| = {abs(f1 - f2)}")
print(f"  f1 + f2 = {f1 + f2}")
print(f"  N - (f1 + f2) = {N - (f1 + f2)}")
print(f"  N - |f1 - f2| = {N - abs(f1 - f2)}")