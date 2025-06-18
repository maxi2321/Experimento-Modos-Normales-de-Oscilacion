import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ==== PARÁMETROS ====
archivo_csv = "/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/experimento_20250602_151642/trayectorias.csv"  # <- Cambiar si es otro archivo
columna_tiempo = "tiempo_s"
columnas_analizar = ["x_rojo", "y_rojo", "x_azul", "y_azul"]
carpeta_salida = "resultados_fft"
os.makedirs(carpeta_salida, exist_ok=True)

# ==== CARGAR DATOS ====
df = pd.read_csv(archivo_csv)
t = df[columna_tiempo].values
dt = np.mean(np.diff(t))
fs = 1 / dt
N = len(t)

# ==== ANALIZAR CADA SEÑAL ====
for columna in columnas_analizar:
    y = df[columna].values

    # Eliminar nans si hay
    if np.isnan(y).any():
        mask = ~np.isnan(y)
        y = y[mask]
        t_recortado = t[mask]
        N = len(t_recortado)
    else:
        t_recortado = t

    # FFT
    yf = fft(y - np.mean(y))  # quitar componente DC
    xf = fftfreq(N, dt)[:N // 2]
    amplitud = 2.0 / N * np.abs(yf[:N // 2])

    # Frecuencia dominante
    idx_max = np.argmax(amplitud)
    f_dom = xf[idx_max]
    T_dom = 1 / f_dom if f_dom != 0 else np.inf

    # ==== GRAFICAR ====
    plt.figure()
    plt.plot(xf, amplitud, label="FFT")
    plt.axvline(f_dom, color='red', linestyle='--', alpha=0.7)
    plt.text(f_dom, amplitud[idx_max] * 0.9,
             f"f = {f_dom:.2f} Hz\nT = {T_dom:.2f} s",
             color='red', ha='left', va='top', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    plt.title(f"FFT de {columna}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/fft_{columna}.png")
    plt.show()

    print(f"✅ {columna}: Frecuencia dominante = {f_dom:.2f} Hz (T ≈ {T_dom:.2f} s)")