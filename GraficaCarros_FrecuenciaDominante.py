import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import strftime, localtime
from scipy.fft import fft, fftfreq

# ==================== PARÁMETROS ====================
nombre_archivo = "video_de_entrada.avi"  # <<< Cambiá esto por el nombre de tu archivo
fs = 30  # FPS (asegurate que coincida con el del video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Crear carpeta de salida
timestamp_str = strftime("experimento_%Y%m%d_%H%M%S", localtime())
os.makedirs(timestamp_str, exist_ok=True)

# Abrir video desde archivo
cap = cv2.VideoCapture(nombre_archivo)
if not cap.isOpened():
    raise IOError("No se pudo abrir el video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_puntos = cv2.VideoWriter(f"{timestamp_str}/video_con_puntos.avi", fourcc, fs, (width, height))

datos = []
frame_idx = 0

print("🎞 Procesando video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tiempo = frame_idx / fs
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtro rojo
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Filtro azul
    mask_blue = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([130, 255, 255]))

    # Centro rojo
    M_red = cv2.moments(mask_red)
    if M_red["m00"] != 0:
        cx_red = int(M_red["m10"] / M_red["m00"])
        cy_red = int(M_red["m01"] / M_red["m00"])
        cv2.circle(frame, (cx_red, cy_red), 5, (0, 0, 255), -1)
    else:
        cx_red, cy_red = np.nan, np.nan

    # Centro azul
    M_blue = cv2.moments(mask_blue)
    if M_blue["m00"] != 0:
        cx_blue = int(M_blue["m10"] / M_blue["m00"])
        cy_blue = int(M_blue["m01"] / M_blue["m00"])
        cv2.circle(frame, (cx_blue, cy_blue), 5, (255, 0, 0), -1)
    else:
        cx_blue, cy_blue = np.nan, np.nan

    datos.append([tiempo, cx_red, cy_red, cx_blue, cy_blue])
    out_puntos.write(frame)

    frame_idx += 1

cap.release()
out_puntos.release()
print("✅ Video procesado. Generando gráficos...")

# ==================== GUARDAR CSV ====================
df = pd.DataFrame(datos, columns=["tiempo_s", "x_rojo", "y_rojo", "x_azul", "y_azul"])
df.dropna(inplace=True)
df.to_csv(f"{timestamp_str}/trayectorias.csv", index=False)

# ==================== GRAFICAR TRAYECTORIAS ====================
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(df["tiempo_s"], df["x_rojo"], 'r-', label="x carro rojo")
plt.plot(df["tiempo_s"], df["y_rojo"], 'r--', label="y carro rojo")
plt.ylabel("Posición (px)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df["tiempo_s"], df["x_azul"], 'b-', label="x carro azul")
plt.plot(df["tiempo_s"], df["y_azul"], 'b--', label="y carro azul")
plt.ylabel("Posición (px)")
plt.xlabel("Tiempo (s)")
plt.legend()

plt.tight_layout()
plt.savefig(f"{timestamp_str}/grafico_trayectorias.png")
plt.show()

# ==================== FFT PARA FRECUENCIA DOMINANTE ====================
def calcular_fft(signal, fs):
    N = len(signal)
    yf = fft(signal - np.mean(signal))
    xf = fftfreq(N, 1 / fs)
    idx = xf > 0
    return xf[idx], np.abs(yf[idx])

frecs_rojo, mag_rojo = calcular_fft(df["x_rojo"].values, fs)
frecs_azul, mag_azul = calcular_fft(df["x_azul"].values, fs)

f_dom_rojo = frecs_rojo[np.argmax(mag_rojo)]
f_dom_azul = frecs_azul[np.argmax(mag_azul)]

# Graficar FFT
plt.figure(figsize=(10, 5))
plt.plot(frecs_rojo, mag_rojo, 'r', label=f"Rojo: {f_dom_rojo:.2f} Hz")
plt.plot(frecs_azul, mag_azul, 'b', label=f"Azul: {f_dom_azul:.2f} Hz")
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"{timestamp_str}/fft_frecuencias.png")
plt.show()

# ==================== RESULTADO FINAL ====================
print(f"🔴 Frecuencia dominante carro rojo: {f_dom_rojo:.3f} Hz")
print(f"🔵 Frecuencia dominante carro azul: {f_dom_azul:.3f} Hz")
print(f"📁 Archivos guardados en: {timestamp_str}")