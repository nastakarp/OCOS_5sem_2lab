# 1_continuous_fourier_properties.py
# Используется угловая частота: F(ω) = ∫ f(t) e^{-i ω t} dt
# Полная символьная проверка + графическая интерпретация

import sympy as sp
from sympy import integrate, exp, I, pi, simplify, symbols, oo
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# ==============================
# 1. СИМВОЛЬНАЯ ПРОВЕРКА СВОЙСТВ
# ==============================

# Символы
t, w, t0, w0, k = symbols('t w t0 w0 k', real=True)
a, b = symbols('a b', complex=True)

# Кастомное преобразование Фурье с угловой частотой
def fourier_transform_angular(f_expr, t, w):
    return integrate(f_expr * exp(-I * w * t), (t, -oo, oo))

# Тестовая функция (гауссиана)
f_test = exp(-t**2)
F_test = fourier_transform_angular(f_test, t, w)
print("Тестовая функция f(t) =", f_test)
print("Её Фурье-образ F(ω) =", simplify(F_test))
print("-" * 60)

# 1. Линейность
print("1. Линейность:")
g_test = exp(-(t - 1)**2)
expr1 = a * f_test + b * g_test
FT1 = fourier_transform_angular(expr1, t, w)
FT1_expected = a * F_test + b * fourier_transform_angular(g_test, t, w)
print("Равенство выполняется:", simplify(FT1 - FT1_expected) == 0)
print("-" * 60)

# 2. Сдвиг во времени
print("2. Сдвиг во времени:")
f_shifted = f_test.subs(t, t - t0)
FT_shift = fourier_transform_angular(f_shifted, t, w)
FT_shift_expected = F_test * exp(-I * w * t0)
print("Равенство выполняется:", simplify(FT_shift - FT_shift_expected) == 0)
print("-" * 60)

# 3. Модуляция
print("3. Модуляция:")
f_mod = f_test * exp(I * w0 * t)
FT_mod = fourier_transform_angular(f_mod, t, w)
FT_mod_expected = F_test.subs(w, w - w0)
print("Равенство выполняется:", simplify(FT_mod - FT_mod_expected) == 0)
print("-" * 60)

# 4. Масштабирование (k > 0)
print("4. Масштабирование (k > 0):")
k_pos = symbols('k_pos', positive=True)
f_scaled = f_test.subs(t, k_pos * t)
FT_scaled = fourier_transform_angular(f_scaled, t, w)
FT_scaled_expected = (1 / k_pos) * F_test.subs(w, w / k_pos)
print("Равенство выполняется:", simplify(FT_scaled - FT_scaled_expected) == 0)
print("-" * 60)

# 5. Свёртка
print("5. Свёртка:")
tau = symbols('tau')
conv_fg = integrate(f_test.subs(t, t - tau) * g_test.subs(t, tau), (tau, -oo, oo))
FT_conv = fourier_transform_angular(conv_fg, t, w)
FT_prod = F_test * fourier_transform_angular(g_test, t, w)
print("Равенство выполняется:", simplify(FT_conv - FT_prod) == 0)
print("-" * 60)

# 6. Умножение → свёртка
print("6. Умножение во времени → свёртка в частоте:")
print("Принцип верен при использовании угловой частоты: "
      "FT{f(t)g(t)} = (1/(2π)) ∫ F(ν) G(ω - ν) dν")
print("-" * 60)

# 7. Дифференцирование
print("7. Дифференцирование:")
df_dt = sp.diff(f_test, t)
FT_deriv = fourier_transform_angular(df_dt, t, w)
FT_deriv_expected = I * w * F_test
print("Равенство выполняется:", simplify(FT_deriv - FT_deriv_expected) == 0)
print("-" * 60)

# ====================================
# 2. ГРАФИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
# ====================================

print("=== Построение графиков для визуализации свойств ===")

# Параметры численного моделирования
T = 6.0               # интервал времени [-T/2, T/2]
N = 2048              # количество точек
t_num = np.linspace(-T/2, T/2, N, endpoint=False)
dt = t_num[1] - t_num[0]

# Исходный сигнал
f_orig = np.exp(-t_num**2)

def compute_spectrum(signal, dt):
    """Вычисляет амплитудный спектр с угловой частотой."""
    N = len(signal)
    F = fftshift(fft(signal)) * dt / np.sqrt(2*np.pi)
    w_axis = fftshift(fftfreq(N, d=dt)) * 2 * np.pi  # ω = 2πf
    return w_axis, np.abs(F)

w_axis, F_orig_spec = compute_spectrum(f_orig, dt)

# 2. Сдвиг во времени
t0_val = 1.0
f_shifted_num = np.exp(-(t_num - t0_val)**2)
_, F_shifted_spec = compute_spectrum(f_shifted_num, dt)

# 3. Модуляция (реальная: cos)
w0_val = 3.0
f_mod_num = f_orig * np.cos(w0_val * t_num)
_, F_mod_spec = compute_spectrum(f_mod_num, dt)

# 4. Масштабирование (k = 0.5 → растяжение)
k_val = 0.5
f_scaled_num = np.exp(-(k_val * t_num)**2)
_, F_scaled_spec = compute_spectrum(f_scaled_num, dt)

# 7. Дифференцирование
df_dt_num = np.gradient(f_orig, dt)
_, F_deriv_spec = compute_spectrum(df_dt_num, dt)
F_deriv_theory = np.abs(w_axis) * F_orig_spec

# Построение графиков
plt.figure(figsize=(14, 12))

# Исходный
plt.subplot(4, 2, 1)
plt.plot(t_num, f_orig, 'b')
plt.title('Исходный сигнал $f(t) = e^{-t^2}$')
plt.grid(True)

plt.subplot(4, 2, 2)
plt.plot(w_axis, F_orig_spec, 'b')
plt.title('Спектр $|F(\\omega)|$')
plt.grid(True)

# Сдвиг
plt.subplot(4, 2, 3)
plt.plot(t_num, f_shifted_num, 'r', label=f'$t_0 = {t0_val}$')
plt.title('Сдвиг во времени')
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(w_axis, F_orig_spec, 'b--', alpha=0.6)
plt.plot(w_axis, F_shifted_spec, 'r')
plt.title('Спектр после сдвига (амплитуда не изменилась)')
plt.grid(True)
plt.xlim(-6, 6)

# Модуляция
plt.subplot(4, 2, 5)
plt.plot(t_num, f_mod_num, 'g')
plt.title(f'Модуляция: $f(t)\\cos({w0_val} t)$')
plt.grid(True)

plt.subplot(4, 2, 6)
plt.plot(w_axis, F_mod_spec, 'g')
plt.title('Спектр после модуляции (боковые полосы)')
plt.grid(True)
plt.xlim(-6, 6)

# Масштабирование
plt.subplot(4, 2, 7)
plt.plot(t_num, f_scaled_num, 'm', label=f'$k = {k_val}$')
plt.title('Масштабирование во времени (растяжение)')
plt.grid(True)
plt.legend()

plt.subplot(4, 2, 8)
plt.plot(w_axis, F_orig_spec, 'b--', alpha=0.6)
plt.plot(w_axis, F_scaled_spec, 'm')
plt.title('Спектр после масштабирования (сужение)')
plt.grid(True)
plt.xlim(-5, 5)
plt.legend()

plt.tight_layout()
plt.savefig("fourier_properties_time_freq.png", dpi=150)
plt.show()

# График дифференцирования отдельно
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t_num, df_dt_num, 'c')
plt.title('Численная производная $df/dt$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(w_axis, F_deriv_spec, 'c', label='Численный спектр')
plt.plot(w_axis, F_deriv_theory, 'k--', label='$|\\omega| \\cdot |F(\\omega)|$')
plt.title('Спектр производной')
plt.grid(True)
plt.legend()
plt.xlim(-5, 5)
plt.tight_layout()
plt.savefig("fourier_differentiation.png", dpi=150)
plt.show()

print("Графики сохранены: fourier_properties_time_freq.png, fourier_differentiation.png")