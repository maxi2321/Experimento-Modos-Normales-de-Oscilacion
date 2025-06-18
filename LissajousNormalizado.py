import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
archivo_csv = ('/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/MasasIgualesPlasticosModoRapido/trayectorias.csv')
Fs = 16  # Tamaño de fuente

# ------------------------------
# CARGA DE DATOS
# ------------------------------
df = pd.read_csv(archivo_csv).dropna()
t = df['tiempo_s'].to_numpy()
x1_raw = df['x_rojo'].to_numpy().astype(float)
x2_raw = df['x_azul'].to_numpy().astype(float)

# Centrado y normalización con el máximo absoluto
x1_centrada = x1_raw - np.mean(x1_raw)
x2_centrada = x2_raw - np.mean(x2_raw)

x1 = x1_centrada / np.max(np.abs(x1_centrada))
x2 = x2_centrada / np.max(np.abs(x2_centrada))

# ------------------------------
# CREAR CARPETA DE SALIDA
# ------------------------------
carpeta_base = os.path.dirname(os.path.abspath(archivo_csv))
carpeta_salida = os.path.join(carpeta_base, 'Lissajous')
os.makedirs(carpeta_salida, exist_ok=True)

# ------------------------------
# GRÁFICO LISSAJOUS NORMALIZADO POR MÁXIMO
# ------------------------------
plt.figure(figsize=(6, 6))
sc = plt.scatter(x1, x2, c=t, cmap='inferno', s=10)
cbar = plt.colorbar(sc)
cbar.set_label('t [s]', fontsize=Fs)
cbar.ax.tick_params(labelsize=Fs)
plt.xlabel('x1', fontsize=Fs)
plt.ylabel('x2', fontsize=Fs)
plt.tick_params(axis='both', labelsize=Fs)
plt.axis('equal')
plt.tight_layout()

# Guardar imagen
plt.savefig(os.path.join(carpeta_salida, 'lissajous_normalizado_maximo_todo_el_intervalo.png'))
plt.close()
