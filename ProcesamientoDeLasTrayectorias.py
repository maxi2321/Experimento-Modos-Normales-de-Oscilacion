import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ------------------------------
# 📁 Cargar datos
# ------------------------------
archivo_csv = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/MasasIgualesPlasticosaModoLento/trayectorias.csv'
df = pd.read_csv(archivo_csv)

t = df["tiempo_s"].values
x_azul = df["x_azul"].values
x_rojo = df["x_rojo"].values

# ------------------------------
# 🗂 Crear carpeta para guardar figuras
# ------------------------------
csv_dir = os.path.dirname(os.path.abspath(archivo_csv))
carpeta_salida = os.path.join(csv_dir, "figuras_fft")
os.makedirs(carpeta_salida, exist_ok=True)

# ------------------------------
# ⚙️ Función FFT sin frecuencia 0
# ------------------------------
def analizar_fft(x, t):
    N = len(t)
    dt = t[1] - t[0]
    freqs = fftfreq(N, dt)
    X = fft(x)
    
    # Parte positiva y quitamos la frecuencia 0
    freqs_pos = freqs[1:N//2]
    mags_pos = np.abs(X)[1:N//2]

    f_dom = freqs_pos[np.argmax(mags_pos)]
    return freqs_pos, mags_pos, f_dom

# ------------------------------
# 🔵 Carro azul: x(t)
# ------------------------------
plt.figure(figsize=(8, 4))
plt.plot(t, x_azul, label="x azul", color="blue")
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición x [pixeles]")
plt.title("Carro Azul - x(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
ruta_azul_tray = os.path.join(carpeta_salida, "carro_azul_trayectoria.png")
plt.savefig(ruta_azul_tray)
plt.close()

# 🔵 Carro azul: FFT
freqs_azul, mag_azul, f_dom_azul = analizar_fft(x_azul, t)

plt.figure(figsize=(8, 4))
plt.plot(freqs_azul, mag_azul, label=f"FFT azul (f = {f_dom_azul:.3f} Hz)", color="blue")
plt.axvline(f_dom_azul, color="blue", linestyle="--")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("Carro Azul - FFT(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
ruta_azul_fft = os.path.join(carpeta_salida, "carro_azul_fft.png")
plt.savefig(ruta_azul_fft)
plt.close()

# ------------------------------
# 🔴 Carro rojo: x(t)
# ------------------------------
plt.figure(figsize=(8, 4))
plt.plot(t, x_rojo, label="x rojo", color="red")
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición x [pixeles]")
plt.title("Carro Rojo - x(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
ruta_rojo_tray = os.path.join(carpeta_salida, "carro_rojo_trayectoria.png")
plt.savefig(ruta_rojo_tray)
plt.close()

# 🔴 Carro rojo: FFT
freqs_rojo, mag_rojo, f_dom_rojo = analizar_fft(x_rojo, t)

plt.figure(figsize=(8, 4))
plt.plot(freqs_rojo, mag_rojo, label=f"FFT rojo (f = {f_dom_rojo:.3f} Hz)", color="red")
plt.axvline(f_dom_rojo, color="red", linestyle="--")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("Carro Rojo - FFT(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
ruta_rojo_fft = os.path.join(carpeta_salida, "carro_rojo_fft.png")
plt.savefig(ruta_rojo_fft)
plt.close()