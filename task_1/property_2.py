import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 2048
T = 12.0                     # общая длина временного окна
dt = T / N
t = np.linspace(-T/2, T/2, N, endpoint=False)

# Параметры гаусса и сдвига
sigma = 0.8
t0 = 2.0                     # сдвиг вправо на 2 единицы

# Гауссова функция (центрирована в 0)
f = np.exp(-0.5 * (t / sigma)**2)

# Сдвинутая версия: f(t - t0) → центр в t0
f_shift = np.exp(-0.5 * ((t - t0) / sigma)**2)

# FFT с корректным центрированием
def fft_centered(signal, dt):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal))) * dt

F_orig = fft_centered(f, dt)           # F(ω)
F_direct = fft_centered(f_shift, dt)   # ℱ{f(t - t0)}

# Частотная ось (угловая частота ω)
df = 1.0 / T
freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi  # ω = 2πf

# Теоретический спектр с учётом сдвига
F_theory = F_orig * np.exp(-1j * freq * t0)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# 1. Временная область
axs[0].plot(t, f, label=r'$f(t)$', linewidth=2)
axs[0].plot(t, f_shift, '--', label=r'$f(t - t_0)$, $t_0 = %.1f$' % t0, linewidth=2)
axs[0].set_xlabel('Время $t$')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('Сигналы во временной области')
axs[0].legend()
axs[0].grid(True)

# 2. Амплитудный спектр
axs[1].plot(freq, np.abs(F_direct), 'b', label=r'$|\mathcal{F}\{f(t - t_0)\}|$', linewidth=2)
axs[1].plot(freq, np.abs(F_theory), 'r--', label=r'$|F(\omega) e^{-i\omega t_0}|$', linewidth=1.5)
axs[1].set_xlabel('Угловая частота $\\omega$')
axs[1].set_ylabel('Амплитуда')
axs[1].set_title('Амплитудный спектр')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(-10, 10)

# 3. Фазовый спектр
axs[2].plot(freq, np.angle(F_direct), 'b', label=r'$\angle \mathcal{F}\{f(t - t_0)\}$', linewidth=2)
axs[2].plot(freq, np.angle(F_theory), 'r--', label=r'$\angle (F(\omega) e^{-i\omega t_0})$', linewidth=1.5)
axs[2].set_xlabel('Угловая частота $\\omega$')
axs[2].set_ylabel('Фаза (рад)')
axs[2].set_title('Фазовый спектр (линейный сдвиг)')
axs[2].legend()
axs[2].grid(True)
axs[2].set_xlim(-10, 10)

plt.tight_layout()
plt.show()

# --- Численная проверка точности ---
diff_abs = np.abs(F_direct - F_theory)
diff_phase = np.angle(F_direct / F_theory)  # разность фаз

print(f"Макс. ошибка в комплексной плоскости: {diff_abs.max():.2e}")
print(f"Макс. ошибка в фазе (рад):           {np.abs(diff_phase).max():.2e}")