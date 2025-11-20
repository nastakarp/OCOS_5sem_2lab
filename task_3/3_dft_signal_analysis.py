import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Параметры ---
Fs = 5000  # Частота дискретизации, Гц
files_to_analyze = [
    'Hz_100.csv',
    'Hz_2500.csv',
    'Hz_5000.csv',
    'Hz_10000.csv'
]

# Создаём папку для сохранения графиков (если не существует)
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)


def analyze_and_save_signal(filename):
    print(f"\n=== АНАЛИЗ ФАЙЛА: {filename} ===")

    # Загрузка данных
    try:
        data = pd.read_csv(filename, header=None).values.flatten()
    except FileNotFoundError:
        print(f"Файл '{filename}' не найден. Пропускаем.")
        return

    N = len(data)
    T = N / Fs
    print(f"Отсчётов: {N}, Длительность: {T:.4f} с")

    # Удаление DC
    data = data - np.mean(data)

    # БПФ
    X = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, 1 / Fs)
    ampl = np.abs(X) / N
    phase = np.angle(X)

    # Только положительные частоты
    half = N // 2
    freqs = freqs[:half]
    ampl = ampl[:half] * 2
    phase = phase[:half]

    # Оценка основной гармоники (максимум, исключая DC)
    idx_max = np.argmax(ampl[1:]) + 1
    f_est = freqs[idx_max]
    A_est = ampl[idx_max]
    phi_est = phase[idx_max]

    # После вычисления freqs
    print(f"Первые 10 частот: {freqs[:10]}")
    print(f"Последние 10 частот: {freqs[-10:]}")

    # Извлечение теоретической частоты из имени файла
    if filename.startswith('Hz_') and filename.endswith('.csv'):
        try:
            f_theory_str = filename.split('_')[1].split('.')[0]
            f_theory = float(f_theory_str)
        except:
            f_theory = None
            print("Не удалось извлечь теоретическую частоту из имени файла.")
    else:
        f_theory = None

    # Вывод результатов
    print(f"Оценка основной гармоники:")
    print(f"  Частота:       {f_est:.2f} Гц")
    print(f"  Амплитуда:     {A_est:.3f} у.е.")
    if f_theory is not None:
        print(f"  Теория:        {f_theory} Гц")
        delta_f = abs(f_est - f_theory)
        print(f"  Расхождение:   {delta_f:.2f} Гц")
    else:
        print("  Теория:        не определена")

    # Интерпретация фазы
    if abs(phi_est) < 0.2:
        phase_desc = "начинается как синус (с нуля и возрастает)"
    elif abs(abs(phi_est) - np.pi / 2) < 0.2:
        phase_desc = "начинается как косинус (с максимума)"
    elif abs(abs(phi_est) - np.pi) < 0.2:
        phase_desc = "начинается с минимума (инвертирован)"
    else:
        phase_desc = "произвольное начальное смещение"
    print(f"  Фаза:          {phi_est:.3f} рад ({np.degrees(phi_est):.1f}°) → {phase_desc}")

    # --- Построение и сохранение графиков ---
    t = np.arange(N) / Fs

    # График 1: Временной сигнал
    plt.figure(figsize=(10, 3))
    plt.plot(t, data, linewidth=0.7)
    plt.title(f'Временной сигнал: {filename}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename.replace('.csv', '')}_time.png"), dpi=150)
    plt.close()

    # График 2: Амплитудный спектр (до 500 Гц)
    plt.figure(figsize=(10, 3))
    plt.plot(freqs, ampl)
    plt.title(f'Амплитудный спектр: {filename}')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(0, 500)
    plt.ylim(0, max(ampl) * 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename.replace('.csv', '')}_spectrum_full.png"), dpi=150)
    plt.close()

    # График 3: Амплитудный спектр вблизи ожидаемой частоты
    if f_theory is not None:
        zoom_range = 50
        xlim_min = max(0, f_theory - zoom_range)
        xlim_max = f_theory + zoom_range
    else:
        zoom_range = 50
        xlim_min = max(0, f_est - zoom_range)
        xlim_max = f_est + zoom_range

    plt.figure(figsize=(10, 3))
    plt.stem(freqs, ampl, basefmt=" ", linefmt='C1-', markerfmt='C1o')
    plt.title(f'Амплитудный спектр: {filename}')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(xlim_min, xlim_max)

    # Безопасное определение ylim
    mask = (freqs >= xlim_min) & (freqs <= xlim_max)
    visible_ampl = ampl[mask]
    if len(visible_ampl) > 0:
        plt.ylim(0, max(visible_ampl) * 1.1)
    else:
        plt.ylim(0, max(ampl) * 1.1)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename.replace('.csv', '')}_spectrum_zoom.png"), dpi=150)
    plt.close()

    print(f"Графики сохранены в папку: {save_dir}")


# --- Основной цикл по всем файлам ---
for fname in files_to_analyze:
    analyze_and_save_signal(fname)

print("\nАнализ всех файлов завершён!")

# ===============================
# Часть 1: Синтетический комплексный сигнал (1 Гц, Fs=10 Гц)
# ===============================
def analyze_synthetic_complex_signal():
    print("\n" + "="*60)
    print("АНАЛИЗ СИНТЕТИЧЕСКОГО КОМПЛЕКСНОГО СИГНАЛА (1 Гц, Fs=10 Гц)")
    print("="*60)

    # Параметры сигнала
    Fs = 10          # Гц
    f0 = 1           # Гц
    duration = 1.0   # с
    N = int(Fs * duration)  # = 10 отсчётов

    n = np.arange(N)
    x = np.exp(1j * 2 * np.pi * f0 * n / Fs)  # x[n] = e^{i 2π f0 n / Fs}

    print(f"Создан комплексный сигнал: x[n] = exp(i·2π·{f0}·n/{Fs})")
    print(f"Длина: {N} отсчётов, частота дискретизации: {Fs} Гц")

    # --- 1. Исходный ДПФ (N=10) ---
    X_orig = np.fft.fft(x)
    freqs_orig = np.fft.fftfreq(N, d=1/Fs)

    # --- 2. Zero-padding до 200 отсчётов ---
    N_pad = 200
    x_padded = np.concatenate([x, np.zeros(N_pad - N)])
    X_padded = np.fft.fft(x_padded)
    freqs_padded = np.fft.fftfreq(N_pad, d=1/Fs)

    # --- 3. Оконное взвешивание (Хэмминг) + zero-padding до 200 ---
    window = np.hamming(N)
    x_windowed = x * window
    x_windowed_padded = np.concatenate([x_windowed, np.zeros(N_pad - N)])
    X_windowed = np.fft.fft(x_windowed_padded)
    # freqs_padded уже есть

    # --- Построение графиков ---
    save_dir = "./"
    os.makedirs(save_dir, exist_ok=True)

    # Общие настройки для спектров (только положительные частоты, т.к. комплексный сигнал — нет симметрии)
    def plot_complex_spectrum(ax_mag, ax_phase, freqs, X, title, highlight_bin=None):
        # Для комплексного сигнала показываем весь диапазон [-Fs/2, Fs/2] или [0, Fs)
        # Но для наглядности — сдвигаем в [-Fs/2, Fs/2)
        X_shifted = np.fft.fftshift(X)
        freqs_shifted = np.fft.fftshift(freqs)

        ax_mag.plot(freqs_shifted, np.abs(X_shifted), 'b-')
        ax_phase.plot(freqs_shifted, np.angle(X_shifted), 'r-')
        ax_mag.set_title(f'{title} — амплитуда')
        ax_phase.set_title(f'{title} — фаза')
        ax_mag.set_xlabel('Частота (Гц)')
        ax_phase.set_xlabel('Частота (Гц)')
        ax_mag.grid(True)
        ax_phase.grid(True)
        ax_mag.set_ylabel('|X(f)|')
        ax_phase.set_ylabel('∠X(f), рад')

        # Отметим теоретическую частоту
        ax_mag.axvline(f0, color='k', linestyle='--', linewidth=1, label=f'Теория: {f0} Гц')
        ax_phase.axvline(f0, color='k', linestyle='--', linewidth=1)
        ax_mag.legend()

    # Создаём фигуру с 3 строками (оригинал, zero-pad, окно)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle('Спектры комплексного гармонического сигнала (1 Гц)', fontsize=14)

    plot_complex_spectrum(axes[0, 0], axes[0, 1], freqs_orig, X_orig, 'Исходный сигнал (N=10)')
    plot_complex_spectrum(axes[1, 0], axes[1, 1], freqs_padded, X_padded, 'Zero-padding до N=200')
    plot_complex_spectrum(axes[2, 0], axes[2, 1], freqs_padded, X_windowed, 'Окно Хэмминга + zero-pad')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "synthetic_complex_signal_analysis.png"), dpi=150)
    plt.close()

    # --- Вывод интерпретации ---
    print("\nИнтерпретация:")
    print("1. Исходный ДПФ (N=10):")
    print("   → Частота 1 Гц точно совпадает с бином ДПФ (k=1), поэтому спектр — один ненулевой отсчёт при f=1 Гц.")
    print("   → Амплитуда = N = 10, фаза ≈ 0.")

    print("\n2. Zero-padding до 200:")
    print("   → Не добавляет новой информации, но интерполирует спектр.")
    print("   → Позволяет «видеть» форму пика более гладко (в данном случае — идеальный пик, остальные ~0).")

    print("\n3. Оконная функция (Хэмминг):")
    print("   → Хотя частота совпадает с бином, окно всё равно «размазывает» спектр (уменьшает амплитуду, добавляет боковые лепестки).")
    print("   → Это демонстрирует общий эффект окон: подавление утечки при несовпадении частот, но ухудшение разрешения.")

    print("\n4. Сравнение с непрерывным спектром:")
    print("   → Непрерывный комплексный тон имеет спектр в виде дельта-функции δ(f - f₀).")
    print("   → ДПФ даёт дискретную выборку этого спектра. При совпадении частоты — получаем эквивалент дельты в одном бине.")

    print("\nГрафик сохранён: synthetic_complex_signal_analysis.png")


# Запуск анализа синтетического сигнала
analyze_synthetic_complex_signal()

print("\nВесь анализ (включая синтетический сигнал) завершён!")
