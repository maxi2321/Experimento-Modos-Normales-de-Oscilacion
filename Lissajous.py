import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
archivo_csv = ('/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/MasasIgualesPlasticosaModoLento/trayectorias.csv')
t_min = 0      # segundos
t_max = 60  # segundos
Fs = 16        # Tamaño de fuente

# ------------------------------
# CARGA DE DATOS
# ------------------------------
df = pd.read_csv(archivo_csv).dropna()
t_full = df['tiempo_s'].to_numpy()
x_rojo = df['x_rojo'].to_numpy().astype(float)
x_azul = df['x_azul'].to_numpy().astype(float)

# Aplicar máscara de tiempo
mask = (t_full >= t_min) & (t_full <= t_max)
t = t_full[mask]
x1 = x_rojo[mask] - np.mean(x_rojo[mask])
x2 = x_azul[mask] - np.mean(x_azul[mask])

# ------------------------------
# CREAR CARPETA DE SALIDA
# ------------------------------
carpeta_base = os.path.dirname(os.path.abspath(archivo_csv))
carpeta_salida = os.path.join(carpeta_base, 'Lissajous')
os.makedirs(carpeta_salida, exist_ok=True)

# ------------------------------
# GRÁFICO LISSAJOUS COLOREADO
# ------------------------------
plt.figure(figsize=(6, 6))
plt.plot(x1, x2, color='Red', linewidth=0.8, alpha=0.6)  # Línea gris detrás
sc = plt.scatter(x1, x2, c=t, cmap='inferno', s=10)
cbar = plt.colorbar(sc)
cbar.set_label('t [s]', fontsize=Fs)
cbar.ax.tick_params(labelsize=Fs)
plt.xlabel('x1 [cm]', fontsize=Fs)
plt.ylabel('x2 [cm]', fontsize=Fs)
plt.tick_params(axis='both', labelsize=Fs)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'lissajous_coloreado_{t_min:.1f}_a_{t_max:.1f}_s.png'))
plt.close()