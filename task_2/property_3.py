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

# Правая часть: X(k) * Y(k)
product_freq = X * Y

# Левая часть: DFT{ циклическая свёртка }
# Циклическая свёртка = ifft( X * Y )   ← без умножения на N!
conv_cyclic = np.fft.ifft(product_freq)  # это и есть (x * y)(n) — циклическая свёртка

# Теперь берём ДПФ от этой свёртки
left_side = np.fft.fft(conv_cyclic)

# Проверка
print("DFT{циклическая свёртка} == X·Y ?", np.allclose(left_side, product_freq, atol=1e-12))
# → Должно быть True!

# === Альтернатива: вычислить свёртку вручную (для наглядности) ===
def cyclic_convolution(x, y):
    N = len(x)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        for m in range(N):
            result[n] += x[m] * y[(n - m) % N]
    return result

conv_manual = cyclic_convolution(x, y)
left_manual = np.fft.fft(conv_manual)
print("Ручная свёртка совпадает с ifft(X*Y)?", np.allclose(conv_manual, conv_cyclic, atol=1e-12))
print("DFT{ручная свёртка} == X·Y ?", np.allclose(left_manual, product_freq, atol=1e-12))

# === Функция для очистки фазы ===
def clean_phase(fft_vals, threshold=1e-10):
    amp = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    phase[amp < threshold] = 0.0
    return phase

phase_left = clean_phase(left_side)
phase_right = clean_phase(product_freq)

# === Визуализация ===
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(np.abs(left_side), 'b-', label='|DFT{свёртка}|')
plt.plot(np.abs(product_freq), 'r--', label='|X·Y|')
plt.title('Свойство свёртки: амплитудный спектр')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(phase_left, 'b-', label='Фаза (DFT свёртки)')
plt.plot(phase_right, 'r--', label='Фаза (X·Y)')
plt.title('Свойство свёртки: фазовый спектр')
plt.xlabel('k')
plt.ylabel('Фаза (рад)')
plt.ylim(-np.pi, np.pi)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# === Теоретические частоты ===
print(f"\nОжидаемые пики на: {abs(f1-f2)}, {f1+f2}, {N-(f1+f2)}, {N-abs(f1-f2)}")