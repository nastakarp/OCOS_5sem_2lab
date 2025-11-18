import numpy as np
import matplotlib.pyplot as plt

N = 4097
dt = 0.005
t = np.linspace(-20, 20, N)

a, b = 1.0, 1.5
f = np.exp(-a * t**2)
g = np.exp(-b * t**2)
fg = f * g

# Прямой Фурье-образ
H_direct = np.fft.fftshift(np.fft.fft(fg)) * dt

# Фурье-образы
F = np.fft.fftshift(np.fft.fft(f)) * dt
G = np.fft.fftshift(np.fft.fft(g)) * dt

# Ручная свёртка с центрированием
from scipy.signal import fftconvolve
conv_FG = fftconvolve(F, G, mode='same') * (2 * np.pi / (N * dt))  # domega
H_conv = conv_FG / (2 * np.pi)

omega = np.fft.fftshift(np.fft.fftfreq(N, d=dt)) * 2 * np.pi

diff = H_direct - H_conv
print("Макс. разница:", np.max(np.abs(diff)))

plt.figure(figsize=(12, 10))
plt.subplot(3,1,1)
plt.plot(t, f, label='f'); plt.plot(t, g, label='g'); plt.plot(t, fg, label='f·g'); plt.legend(); plt.grid()

plt.subplot(3,1,2)
plt.plot(omega, H_direct.real, label='FFT(f·g)')
plt.plot(omega, H_conv.real, '--', label='(1/2π)(F*G)')
plt.legend(); plt.grid()

plt.subplot(3,1,3)
plt.plot(omega, diff.real); plt.title('Разница'); plt.grid()
plt.show()