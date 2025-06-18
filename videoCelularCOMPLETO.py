import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import strftime, localtime
from scipy.fft import fft, fftfreq

# ==================== PARÁMETROS ====================
fs = 30  # FPS
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# ==================== FASE 1: GRABAR VIDEO SIN PROCESAMIENTO ====================
# Crear carpeta del experimento
timestamp_str = strftime("experimento_%Y%m%d_%H%M%S", localtime())
os.makedirs(timestamp_str, exist_ok=True)

# Captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se pudo acceder a la cámara")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

out_original = cv2.VideoWriter(f"{timestamp_str}/video_original.avi", fourcc, fs, (width, height))
cv2.namedWindow("Vista en vivo", cv2.WINDOW_NORMAL)

frames = []

print("🎥 Grabando video. Presioná 'q' para terminar...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame.copy())
    out_original.write(frame)
    cv2.imshow("Vista en vivo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_original.release()
cv2.destroyAllWindows()
print("✅ Video original guardado. Procesando datos...")

# ==================== FASE 2: PROCESAR VIDEO ====================
datos = []
out_puntos = cv2.VideoWriter(f"{timestamp_str}/video_con_puntos.avi", fourcc, fs, (width, height))

for frame_count, frame in enumerate(frames):
    timestamp = frame_count / fs
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

    datos.append([timestamp, cx_red, cy_red, cx_blue, cy_blue])
    out_puntos.write(frame)

out_puntos.release()

# ==================== GUARDAR CSV ====================
df_final = pd.DataFrame(datos, columns=["tiempo_s", "x_rojo", "y_rojo", "x_azul", "y_azul"])
df_final.dropna(inplace=True)
df_final.to_csv(f"{timestamp_str}/trayectorias.csv", index=False)

# ==================== GRAFICAR TRAYECTORIAS ====================
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(df_final["tiempo_s"], df_final["x_rojo"], 'r-', label="x carro rojo")
plt.plot(df_final["tiempo_s"], df_final["y_rojo"], 'r--', label="y carro rojo")
plt.ylabel("Posición (px)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_final["tiempo_s"], df_final["x_azul"], 'b-', label="x carro azul")
plt.plot(df_final["tiempo_s"], df_final["y_azul"], 'b--', label="y carro azul")
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
    idx_positivos = xf > 0
    return xf[idx_positivos], np.abs(yf[idx_positivos])

x_rojo = df_final["x_rojo"].values
x_azul = df_final["x_azul"].values

frecs_rojo, mag_rojo = calcular_fft(x_rojo, fs)
frecs_azul, mag_azul = calcular_fft(x_azul, fs)

f_dom_rojo = frecs_rojo[np.argmax(mag_rojo)]
f_dom_azul = frecs_azul[np.argmax(mag_azul)]

# --- Graficar FFT
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

# ==================== IMPRIMIR RESULTADOS ====================
print(f"🔴 Frecuencia dominante carro rojo: {f_dom_rojo:.3f} Hz")
print(f"🔵 Frecuencia dominante carro azul: {f_dom_azul:.3f} Hz")
print(f"✅ Todo guardado en la carpeta: {timestamp_str}")