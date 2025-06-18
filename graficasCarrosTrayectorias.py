import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Parámetros ajustables ===
ruta_csv = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/2ResortesMetalicos1ResortePlastico/trayectorias.csv'
tam_puntos = 2                     # Tamaño de los puntos (markersize)
limite_tiempo = (0,50)                 # (t_min, t_max) o None para todo el rango
paso_muestreo = 1                     # Usar 1 para todos los puntos, >1 para submuestrear
# === Figsize === *
a = 12
b = 4
# === Leer archivo CSV ===
df = pd.read_csv(ruta_csv)

# === Recorte de tiempo si se solicita ===
if limite_tiempo is not None:
    t_min, t_max = limite_tiempo
    df = df[(df['tiempo_s'] >= t_min) & (df['tiempo_s'] <= t_max)]

# === Submuestreo si paso_muestreo > 1 ===
df = df.iloc[::paso_muestreo]

# === Extraer y centrar datos ===
t = df['tiempo_s'].values
x_rojo = df['x_rojo'].values - np.mean(df['x_rojo'].values)
x_azul = df['x_azul'].values - np.mean(df['x_azul'].values)

# === Carpeta de salida ===
carpeta_salida = os.path.join(os.path.dirname(ruta_csv), "graficas_centradas")
os.makedirs(carpeta_salida, exist_ok=True)

# === Gráfico carrito rojo ===
plt.figure(figsize=(a, b))
plt.plot(t, x_rojo, 'o-', color='red', markersize=tam_puntos, label='Carrito Rojo')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición centrada [px]')
plt.title('Oscilación centrada - Carrito Rojo')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, 'carrito_rojo_centrado.png'))
plt.show()

# === Gráfico carrito azul ===
plt.figure(figsize=(a, b))
plt.plot(t, x_azul, 'o-', color='blue', markersize=tam_puntos, label='Carrito Azul')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición centrada [px]')
plt.title('Oscilación centrada - Carrito Azul')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, 'carrito_azul_centrado.png'))
plt.show()