import numpy as np
import matplotlib.pyplot as plt

# Параметры дискретизации
N = 2048
T = 10.0                     # общая длительность (в условных единицах)
dt = T / N
t = np.linspace(-T/2, T/2, N, endpoint=False)  # центрируем вокруг 0

# Константы линейной комбинации
a = 1.8
b = -0.7

# Две гауссовы функции
def gaussian(t, mu=0.0, sigma=1.0, amp=1.0):
    return amp * np.exp(-0.5 * ((t - mu) / sigma)**2)

f = gaussian(t, mu=-1.0, sigma=0.5, amp=1.0)   # узкий, сдвинут влево
g = gaussian(t, mu=1.5, sigma=1.2, amp=1.0)   # шире, сдвинут вправо

# Линейная комбинация во временной области
h = a * f + b * g

# FFT (с учётом центрирования)
F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dt
G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dt
H_direct = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(h))) * dt
H_linear = a * F + b * G

# Частотная ось
df = 1.0 / T
freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi  # угловая частота ω

# --- Визуализация ---
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# 1. Временная область
axs[0].plot(t, f, label=r'$f(t)$', alpha=0.7)
axs[0].plot(t, g, label=r'$g(t)$', alpha=0.7)
axs[0].plot(t, h, 'k', linewidth=2, label=r'$a f(t) + b g(t)$')
axs[0].set_xlabel('Время $t$')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('Временная область')
axs[0].legend()
axs[0].grid(True)

# 2. Частотная область: сравнение
axs[1].plot(freq, np.abs(H_direct), 'b', linewidth=2, label=r'$|\mathcal{F}\{a f + b g\}|$')
axs[1].plot(freq, np.abs(H_linear), 'r--', linewidth=1.5, label=r'$|a F(\omega) + b G(\omega)|$')
axs[1].set_xlabel('Угловая частота $\\omega$')
axs[1].set_ylabel('Амплитуда')
axs[1].set_title('Частотная область — проверка линейности')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(-15, 15)

plt.tight_layout()
plt.show()

# --- Проверка разности (для строгости) ---
diff = np.abs(H_direct - H_linear)
print(f"Максимальная разность: {diff.max():.2e}")
print(f"Средняя разность: {diff.mean():.2e}")