import numpy as np
import matplotlib.pyplot as plt

# === Параметры сигнала ===
N = 32                     # число отсчётов
t = np.arange(N)            # дискретное время (в условных единицах)
dt = 1.0                    # шаг по времени (можно считать =1)
# Частоты в циклах на N отсчётов (должны быть целыми для "чистого" спектра!)
f1 = 5                      # частота первого синуса
f2 = 12                     # частота второго синуса

# Сигналы
x = np.sin(2 * np.pi * f1 * t / N)  # sin(2π f1 n / N)
y = np.sin(2 * np.pi * f2 * t / N)  # sin(2π f2 n / N)

# Коэффициенты линейной комбинации
a = 1.5
b = -0.9

# === ДПФ ===
X = np.fft.fft(x)
Y = np.fft.fft(y)

# Левая часть: DFT{a x + b y}
signal_combined = a * x + b * y
left = np.fft.fft(signal_combined)

# Правая часть: a X + b Y
right = a * X + b * Y

# Проверка точности
print("Левая и правая части совпадают?", np.allclose(left, right, atol=1e-12))

# === Функция для "очистки" фазы ===
def clean_phase(fft_vals, threshold=1e-10):
    amp = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    phase[amp < threshold] = 0.0
    return phase

phase_left = clean_phase(left)
phase_right = clean_phase(right)

# === Визуализация ===
plt.figure(figsize=(12, 8))

# Амплитудный спектр
plt.subplot(2, 1, 1)
plt.plot(np.abs(left), 'b-', label='|DFT{a·x + b·y}| (левая)')
plt.plot(np.abs(right), 'r--', label='|a·X + b·Y| (правая)')
plt.title('Амплитудный спектр')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

# Фазовый спектр (очищенный)
plt.subplot(2, 1, 2)
plt.plot(phase_left, 'b-', label='Фаза (левая, очищенная)')
plt.plot(phase_right, 'r--', label='Фаза (правая, очищенная)')
plt.title('Фазовый спектр (фаза занулена при малой амплитуде)')
plt.xlabel('k (номер бина)')
plt.ylabel('Фаза (рад)')
plt.ylim(-np.pi, np.pi)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()