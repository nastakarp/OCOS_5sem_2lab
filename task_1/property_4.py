import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Параметры
N = 4096  # увеличим для лучшей точности при масштабировании
T = 20.0   # широкое временное окно
dt = T / N
t = np.linspace(-T/2, T/2, N, endpoint=False)

# Параметры
sigma = 1.0
k = 2.0  # масштаб: k > 1 → сжатие во времени

# Оригинальный сигнал: гаусс
f = np.exp(-0.5 * (t / sigma)**2)

# Масштабированный сигнал: f(k * t)
f_scaled = np.exp(-0.5 * (k * t / sigma)**2)  # f(k t)

# Вспомогательная функция FFT с нормировкой
def fft_centered(signal, dt):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal))) * dt

# Спектры
F_orig = fft_centered(f, dt)        # F(ω)
F_direct = fft_centered(f_scaled, dt)  # ℱ{f(k t)}

# Частотная ось
freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi

# Теоретический результат: (1/|k|) * F(ω / k)
F_interp = interp1d(freq, F_orig, kind='cubic', fill_value=0, bounds_error=False)
F_theory = (1.0 / abs(k)) * F_interp(freq / k)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# 1. Временная область
axs[0].plot(t, f, 'k', label=r'$f(t)$', linewidth=2)
axs[0].plot(t, f_scaled, 'b', label=r'$f(k t)$, $k = %.1f$' % k, linewidth=2)
axs[0].set_xlabel('Время $t$')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('Масштабирование во временной области')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(-5, 5)

# 2. Амплитудный спектр
axs[1].plot(freq, np.abs(F_direct), 'b', linewidth=2, label=r'$|\mathcal{F}\{f(k t)\}|$')
axs[1].plot(freq, np.abs(F_theory), 'r--', linewidth=1.5, label=r'$\frac{1}{|k|} \left|F\left(\frac{\omega}{k}\right)\right|$')
axs[1].set_xlabel('Угловая частота $\\omega$')
axs[1].set_ylabel('Амплитуда')
axs[1].set_title('Амплитудный спектр')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(-10, 10)

# 3. Разность спектров (логарифмический масштаб)
diff = np.abs(F_direct - F_theory)
axs[2].plot(freq, diff, 'k', linewidth=1.5)
axs[2].set_xlabel('Угловая частота $\\omega$')
axs[2].set_ylabel('Разность')
axs[2].set_title('Разность спектров')
axs[2].set_yscale('log')
axs[2].grid(True)
axs[2].set_xlim(-10, 10)

plt.tight_layout()
plt.show()

# --- Численная проверка ---
print(f"Максимальная разность: {diff.max():.2e}")
print(f"Средняя разность:      {diff.mean():.2e}")