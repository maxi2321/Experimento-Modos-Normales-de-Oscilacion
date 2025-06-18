import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import strftime, localtime
from scipy.fft import fft, fftfreq

def procesar_video(ruta_video):
    # Crear carpeta del experimento
    timestamp_str = strftime("experimento_%Y%m%d_%H%M%S", localtime())
    os.makedirs(timestamp_str, exist_ok=True)

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {ruta_video}")

    # Propiedades de video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30

    datos = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filtro rojo
        lower_red1 = np.array([160 ,100, 100])
        upper_red1 = np.array([180, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                                 cv2.inRange(hsv, lower_red2, upper_red2))

        # Filtro azul
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Centro rojo
        M_red = cv2.moments(mask_red)
        if M_red["m00"] != 0:
            cx_red = int(M_red["m10"] / M_red["m00"])
            cy_red = int(M_red["m01"] / M_red["m00"])
        else:
            cx_red, cy_red = np.nan, np.nan

        # Centro azul
        M_blue = cv2.moments(mask_blue)
        if M_blue["m00"] != 0:
            cx_blue = int(M_blue["m10"] / M_blue["m00"])
            cy_blue = int(M_blue["m01"] / M_blue["m00"])
        else:
            cx_blue, cy_blue = np.nan, np.nan

        datos.append([timestamp, cx_red, cy_red, cx_blue, cy_blue])
        frame_count += 1

    cap.release()

    # Guardar CSV
    df_final = pd.DataFrame(datos, columns=["tiempo_s", "x_rojo", "y_rojo", "x_azul", "y_azul"])
    df_final.to_csv(f"{timestamp_str}/trayectorias.csv", index=False)

    # Graficar trayectorias
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df_final["tiempo_s"], df_final["x_rojo"], '-', label="x carro rojo")
    plt.plot(df_final["tiempo_s"], df_final["y_rojo"], '-', label="y carro rojo")
    plt.ylabel("Posición (px)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(df_final["tiempo_s"], df_final["x_azul"], '-', label="x carro azul")
    plt.plot(df_final["tiempo_s"], df_final["y_azul"], '-', label="y carro azul")
    plt.ylabel("Posición (px)")
    plt.xlabel("Tiempo (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{timestamp_str}/grafico_trayectorias.png")
    plt.show()

    # Análisis FFT
    def analizar_frecuencia(tiempo, posicion, etiqueta, carpeta):
        mask = ~np.isnan(posicion)
        tiempo = tiempo[mask]
        posicion = posicion[mask]
        if len(tiempo) < 2:
            print(f"⚠️ No hay suficientes datos para {etiqueta}")
            return
        dt = np.mean(np.diff(tiempo))
        N = len(posicion)
        yf = fft(posicion - np.mean(posicion))
        xf = fftfreq(N, dt)[:N//2]
        amplitud = 2.0/N * np.abs(yf[0:N//2])
        frecuencia_dominante = xf[np.argmax(amplitud)]
        plt.figure()
        plt.plot(xf, amplitud)
        plt.title(f"FFT de {etiqueta}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.grid()
        f_dom = frecuencia_dominante
        T_dom = 1 / f_dom if f_dom != 0 else np.inf
        amp_dom = np.max(amplitud)
        plt.axvline(f_dom, color='r', linestyle='--', alpha=0.7)
        plt.text(f_dom, amp_dom * 0.9,
                 f"f = {f_dom:.2f} Hz\nT = {T_dom:.2f} s",
                 color='red', ha='left', va='top', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
        plt.savefig(f"{carpeta}/fft_{etiqueta.replace(' ', '_')}.png")
        plt.close()
        print(f"📈 Frecuencia dominante de {etiqueta}: {f_dom:.3f} Hz (T ≈ {T_dom:.3f} s)")

    analizar_frecuencia(df_final["tiempo_s"].values, df_final["x_rojo"].values, "Carro rojo", timestamp_str)
    analizar_frecuencia(df_final["tiempo_s"].values, df_final["x_azul"].values, "Carro azul", timestamp_str)

    print(f"✅ Datos y gráficos guardados en: {timestamp_str}")

# Uso:
ruta_video = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/videosCelularExperimentos/2ResortesMetalicos1ResortePlastico.mp4'
procesar_video(ruta_video)