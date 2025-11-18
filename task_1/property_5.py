import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 2048
dt = 0.01
t = np.linspace(-10, 10, N)
a = 1.0
b = 1.5

f = np.exp(-a * t**2)
g = np.exp(-b * t**2)

# 1. Реальная свёртка (с нормировкой dt)
conv_real = np.convolve(f, g, mode='same') * dt

# 2. Частотная область — без fftshift!
F = np.fft.fft(f)  # Без умножения на dt здесь
G = np.fft.fft(g)

FG = F * G

# 3. Обратное преобразование — умножаем на dt (как в convolve)
conv_freq = np.fft.ifft(FG).real * dt  # IFFT уже делит на N — поэтому не умножаем на N!

# 4. Центрируем результат (как convolve(mode='same'))
conv_freq = np.fft.fftshift(conv_freq)

# Выводим максимальную разницу
print("Максимальная разница:", np.max(np.abs(conv_real - conv_freq)))

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, f, label=r'$f(t) = e^{-a t^2}$', lw=2)
plt.plot(t, g, label=r'$g(t) = e^{-b t^2}$', lw=2)
plt.title('Функции')
plt.xlabel('t')
plt.ylabel('f(t), g(t)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, conv_real, label=r'$(f * g)(t)$ — реальная свёртка', color='blue', linewidth=2)
plt.plot(t, conv_freq, '--', label=r'$\mathcal{F}^{-1}\{F(\omega)G(\omega)\}$ — через частотную область',
         color='red', linewidth=2)

# Опционально: аналитическая свёртка
analytical = np.sqrt(np.pi / (a + b)) * np.exp(- (a * b) / (a + b) * t**2)
plt.title('Сравнение свёрток: реальная vs частотная область')
plt.xlabel('t')
plt.ylabel('Значение свёртки')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(3, 1, 3)
diff = np.abs(conv_real - conv_freq)
plt.plot(t, diff, label='|conv_real - conv_freq|', color='purple')
plt.title('Разница между двумя методами')
plt.xlabel('t')
plt.ylabel('|разница|')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()