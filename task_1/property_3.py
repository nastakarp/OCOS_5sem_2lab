import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 2048
T = 12.0
dt = T / N
t = np.linspace(-T/2, T/2, N, endpoint=False)

# Параметры
sigma = 0.8
omega0 = 4.0  # частота модуляции

# Оригинальный сигнал: гаусс
f = np.exp(-0.5 * (t / sigma)**2)

# Модулированный сигнал: f(t) * exp(i * omega0 * t)
f_mod = f * np.exp(1j * omega0 * t)

# Вспомогательная функция FFT с центрированием
def fft_centered(signal, dt):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal))) * dt

# Спектры
F_orig = fft_centered(f, dt)           # F(ω)
F_direct = fft_centered(f_mod, dt)     # ℱ{f(t) e^{iω0 t}}

# Частотная ось (угловая частота ω)
freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi

# Теоретический сдвиг: F(ω - ω0)
# Нужно интерполировать F_orig на сдвинутой сетке
from scipy.interpolate import interp1d

# Создаём интерполятор для F_orig
F_interp = interp1d(freq, F_orig, kind='cubic', fill_value=0, bounds_error=False)
F_shifted_theory = F_interp(freq - omega0)

# --- Визуализация ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# 1. Временная область (вещественная часть для наглядности)
axs[0].plot(t, f, 'k', label=r'$f(t)$ — огибающая', linewidth=2)
axs[0].plot(t, f_mod.real, 'b', label=r'$\operatorname{Re}\{f(t) e^{i\omega_0 t}\}$', alpha=0.8)
axs[0].set_xlabel('Время $t$')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('Модуляция во временной области')
axs[0].legend()
axs[0].grid(True)

# 2. Амплитудный спектр
axs[1].plot(freq, np.abs(F_direct), 'b', linewidth=2, label=r'$|\mathcal{F}\{f(t)e^{i\omega_0 t}\}|$')
axs[1].plot(freq, np.abs(F_shifted_theory), 'r--', linewidth=1.5, label=r'$|F(\omega - \omega_0)|$')
axs[1].axvline(omega0, color='gray', linestyle=':', label=r'$\omega_0$')
axs[1].set_xlabel('Угловая частота $\\omega$')
axs[1].set_ylabel('Амплитуда')
axs[1].set_title('Амплитудный спектр — сдвиг на $\\omega_0$')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(-10, 10)

# 3. Фазовый спектр
axs[2].plot(freq, np.angle(F_direct), 'b', linewidth=2, label=r'$\angle \mathcal{F}\{f(t)e^{i\omega_0 t}\}$')
axs[2].plot(freq, np.angle(F_shifted_theory), 'r--', linewidth=1.5, label=r'$\angle F(\omega - \omega_0)$')
axs[2].set_xlabel('Угловая частота $\\omega$')
axs[2].set_ylabel('Фаза (рад)')
axs[2].set_title('Фазовый спектр')
axs[2].legend()
axs[2].grid(True)
axs[2].set_xlim(-10, 10)



plt.tight_layout()
plt.show()
# --- Численная проверка ---
diff = np.abs(F_direct - F_shifted_theory)
print(f"Максимальная разность: {diff.max():.2e}")
print(f"Средняя разность:      {diff.mean():.2e}")