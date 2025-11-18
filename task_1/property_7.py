import numpy as np
import matplotlib.pyplot as plt

# Параметры (в стиле первого кода)
N = 4097                # нечётное число точек → симметрия вокруг t=0
T = 20.0                # половина интервала (от -T до +T)
t = np.linspace(-T, T, N)  # включая оба конца (endpoint=True по умолчанию)
dt = t[1] - t[0]

# Исходная функция и её аналитическая производная
f_t = np.exp(-t**2)
df_dt_analytical = -2 * t * np.exp(-t**2)

# Прямое Фурье-преобразование (с масштабированием dt)
F_w = np.fft.fftshift(np.fft.fft(f_t)) * dt

# Частотная ось (рад/с)
omega = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi

# iω F(ω)
iwF_w = 1j * omega * F_w

# Обратное преобразование: нужно отменить fftshift и применить ifft
# Важно: обратное преобразование даёт результат без масштаба dt,
# поэтому умножаем на N / (2π) для согласования с непрерывным Фурье?
# Но проще: поскольку F_w = FFT(f) * dt, то f = ifft(F_w/dt) * N
# Однако мы хотим F⁻¹{iωF(ω)} ≈ df/dt, и для самосогласованности:
g_t = np.fft.ifft(np.fft.ifftshift(iwF_w)) * N / (2 * np.pi)

# Альтернативный (и более надёжный) способ — использовать прямую формулу:
# Поскольку f(t) быстро затухает, можно просто взять обратное преобразование
# и не париться с нормировкой, если сравнивать формы (а не абсолютные значения).
# Но для точности оставим масштаб, соответствующий непрерывному преобразованию.
#
# Однако на практике часто проще использовать:
# g_t = np.fft.ifft(np.fft.ifftshift(iwF_w)) * N * dt  # это тоже частая нормировка
#
# Но для гауссианы и симметричного интервала лучше проверить по ошибке.
# Попробуем оба варианта и выберем тот, где ошибка минимальна.

# Попробуем более простой и практически корректный подход без сложной нормировки:
# Так как мы сравниваем с аналитической производной, просто вычислим обратное преобразование
# и возьмём его действительную часть — форма должна совпасть.
g_t_simple = np.fft.ifft(np.fft.ifftshift(iwF_w))

# Для точного соответствия с непрерывным Фурье-преобразованием
# (где F(ω) = ∫ f(t) e^{-iωt} dt, f(t) = 1/(2π) ∫ F(ω) e^{iωt} dω),
# нужно: f(t) ≈ (1/(2π)) * ifft(fftshift(F_w)) * N * dω,
# а dω = 2π / (N*dt) → f(t) ≈ ifft(fftshift(F_w)) * N * dω / (2π) = ifft(...) / dt
#
# Поэтому:
d_omega = 2 * np.pi / (N * dt)
g_t_correct = np.fft.ifft(np.fft.ifftshift(iwF_w)) * d_omega * N / (2 * np.pi)
# Но d_omega * N / (2π) = 1 / dt → g_t_correct = ifft(...) / dt

g_t = np.fft.ifft(np.fft.ifftshift(iwF_w)) / dt

# Теперь сравниваем
error = np.max(np.abs(df_dt_analytical - g_t.real))
print(f"Максимальная разница между аналитической производной и F⁻¹{{iωF(ω)}}: {error:.2e}")

# Построение графиков (3 подграфика, как в первом коде)
plt.figure(figsize=(12, 10))

# Подграфик 1: Исходная функция и её производная
plt.subplot(3, 1, 1)
plt.plot(t, f_t, label=r'$f(t) = e^{-t^2}$', color='blue')
plt.plot(t, df_dt_analytical, label=r"$f'(t) = -2t e^{-t^2}$", color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('Амплитуда')
plt.title('Исходная функция и её аналитическая производная')
plt.grid(True)
plt.legend()

# Подграфик 2: Амплитудные спектры
plt.subplot(3, 1, 2)
plt.plot(omega, np.abs(F_w), label=r'$|F(\omega)|$', color='blue')
plt.plot(omega, np.abs(iwF_w), label=r'$|i\omega F(\omega)|$', color='green', linestyle='--')
plt.xlim(-10, 10)  # ограничим для наглядности
plt.xlabel(r'$\omega$ (рад/с)')
plt.ylabel('Амплитуда спектра')
plt.title('Амплитудные спектры: оригинал и iωF(ω)')
plt.grid(True)
plt.legend()

# Подграфик 3: Сравнение производной и обратного преобразования
plt.subplot(3, 1, 3)
plt.plot(t, df_dt_analytical, label=r"Аналитическая $f'(t)$", color='red')
plt.plot(t, g_t.real, '--', label=r"$\mathrm{Re}\left\{ \mathcal{F}^{-1}\{i\omega F(\omega)\} \right\}$", color='purple')
plt.plot(t, df_dt_analytical - g_t.real, ':', label='Разница', color='black', linewidth=1)
plt.xlabel('t')
plt.ylabel('Амплитуда')
plt.title('Сравнение производной и результата обратного преобразования')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()