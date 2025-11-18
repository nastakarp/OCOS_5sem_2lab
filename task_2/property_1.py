import numpy as np
import matplotlib.pyplot as plt

# === Параметры ===
N = 128  # число отсчётов
n = np.arange(N)

# Параметры первой гауссовой функции
n0_1 = 40      # центр
sigma1 = 5     # ширина
x = np.exp(- (n - n0_1)**2 / (2 * sigma1**2))

# Параметры второй гауссовой функции
n0_2 = 80      # другой центр
sigma2 = 8     # другая ширина
y = np.exp(- (n - n0_2)**2 / (2 * sigma2**2))

# Коэффициенты линейной комбинации
a = 1.2
b = -0.7

# === Вычисление ДПФ ===
X = np.fft.fft(x)
Y = np.fft.fft(y)

# Левая часть: DFT{a x(n) + b y(n)}
signal_combined = a * x + b * y
fft_combined = np.fft.fft(signal_combined)

# Правая часть: a X(k) + b Y(k)
right_side = a * X + b * Y

# === Проверка численного равенства ===
print("Левая и правая части совпадают?", np.allclose(fft_combined, right_side, atol=1e-12))

# === Вспомогательная функция для построения спектров ===
def plot_spectrum_left_and_right(left, right):
    plt.figure(figsize=(14, 6))

    # Амплитудный спектр
    plt.subplot(2, 2, 1)
    plt.plot(np.abs(left), 'b', label='Левая часть')
    plt.title('Амплитудный спектр: DFT{a·x + b·y}')
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(right), 'r--', label='Правая часть')
    plt.title('Амплитудный спектр: a·X + b·Y')
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.grid()
    plt.legend()

    # Фазовый спектр
    plt.subplot(2, 2, 3)
    plt.plot(np.angle(left), 'b', label='Левая часть')
    plt.title('Фазовый спектр: DFT{a·x + b·y}')
    plt.xlabel('k')
    plt.ylabel('phase (rad)')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(np.angle(right), 'r--', label='Правая часть')
    plt.title('Фазовый спектр: a·X + b·Y')
    plt.xlabel('k')
    plt.ylabel('phase (rad)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# === Построение графиков ===
plot_spectrum_left_and_right(fft_combined, right_side)
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.abs(fft_combined), 'b-', label='Левая часть')
plt.plot(np.abs(right_side), 'r--', label='Правая часть')
plt.title('Амплитудный спектр: совмещение')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.angle(fft_combined), 'b-', label='Левая часть')
plt.plot(np.angle(right_side), 'r--', label='Правая часть')
plt.title('Фазовый спектр: совмещение')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()