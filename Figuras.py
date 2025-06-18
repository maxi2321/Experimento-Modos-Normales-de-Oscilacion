import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
import os

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
archivo_csv = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/MasasIgualesPlasticosModoRapido/trayectorias.csv'
nombre = 'MasasIgualesPlasticosModoRapido'
tamano_puntos = 15
epsilon = 0.3  # Tolerancia mínima para diferenciar frecuencias dominantes
t_min = 0      # Tiempo mínimo a graficar (trayectorias)
t_max = 5      # Tiempo máximo a graficar (trayectorias)
Fs = 16        # Fontsize de las gráficas
# ------------------------------

# Leer CSV
df = pd.read_csv(archivo_csv)

# Eliminar filas con NaN
df = df.dropna()

t_full = df['tiempo_s'].to_numpy()
x_rojo_full = df['x_rojo'].to_numpy().astype(float)
x_azul_full = df['x_azul'].to_numpy().astype(float)

# --- FFT y Lissajous: usar TODA la señal ---
x_rojo_centrado = x_rojo_full - np.mean(x_rojo_full)
x_azul_centrado = x_azul_full - np.mean(x_azul_full)

# --- Trayectorias: aplicar filtro por tiempo ---
if t_max is None:
    t_max = t_full[-1]
mask = (t_full >= t_min) & (t_full <= t_max)
t_traj = t_full[mask]
x_rojo_traj = x_rojo_centrado[mask]
x_azul_traj = x_azul_centrado[mask]

# --- Crear carpeta junto al CSV ---
ruta_csv = os.path.abspath(archivo_csv)
carpeta_base = os.path.dirname(ruta_csv)
carpeta_salida = os.path.join(carpeta_base, f'figuras_{nombre}' if nombre else 'figuras_experimento')
os.makedirs(carpeta_salida, exist_ok=True)

# --- Función FFT ---
def obtener_frecuencias_dominantes(y, t, epsilon):
    N = len(t)
    dt = t[1] - t[0]
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, dt)
    amplitudes = np.abs(fft_y)

    indices_validos = np.where(freqs > 0)
    freqs = freqs[indices_validos]
    amplitudes = amplitudes[indices_validos]

    orden = np.argsort(amplitudes)[::-1]
    frecs_dom = []
    for idx in orden:
        f = freqs[idx]
        if all(abs(f - f_existente) > epsilon for f_existente in frecs_dom):
            frecs_dom.append(f)
        if len(frecs_dom) == 2:
            break
    return freqs, amplitudes, frecs_dom

# --- Trayectoria Rojo ---
plt.figure(figsize=(10, 4))
plt.plot(t_traj, x_rojo_traj, color='red', alpha=0.6)
plt.scatter(t_traj, x_rojo_traj, color='red', s=tamano_puntos)
plt.xlabel('t[s]', fontsize=Fs+3)
plt.ylabel('x1[cm]', fontsize=Fs+3)
plt.tick_params(axis='both', labelsize=Fs+2)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'{nombre}_trayectoria_rojo.png'))
plt.close()

# --- Trayectoria Azul ---
plt.figure(figsize=(10, 4))
plt.plot(t_traj, x_azul_traj, color='blue', alpha=0.6)
plt.scatter(t_traj, x_azul_traj, color='blue', s=tamano_puntos)
plt.xlabel('t[s]', fontsize=Fs+2)
plt.ylabel('x2[cm]', fontsize=Fs+2)
plt.tick_params(axis='both', labelsize=Fs+2)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'{nombre}_trayectoria_azul.png'))
plt.close()

# --- FFT Rojo --- #
freqs_r, amp_r, frecs_dom_r = obtener_frecuencias_dominantes(x_rojo_centrado, t_full, epsilon)
amp_r_norm = amp_r / np.max(amp_r)

plt.figure(figsize=(8, 4))
plt.semilogy(freqs_r, amp_r_norm, color='red')
handles = []
labels = []
for f in frecs_dom_r:
    line = plt.axvline(f, color='black', linestyle='--', linewidth=0.8)
    handles.append(line)
    labels.append(f'{f:.2f} Hz')
plt.legend(handles, labels, loc='upper right', fontsize=Fs+1, title_fontsize=Fs+1)
plt.xlabel('f[Hz]', fontsize=Fs+2)
plt.tick_params(axis='both', labelsize=Fs+2)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'{nombre}_fft_rojo.png'))
plt.close()

# --- FFT Azul ---
freqs_b, amp_b, frecs_dom_b = obtener_frecuencias_dominantes(x_azul_centrado, t_full, epsilon)
amp_b_norm = amp_b / np.max(amp_b)

plt.figure(figsize=(8, 4))
plt.semilogy(freqs_b, amp_b_norm, color='blue')
handles = []
labels = []
for f in frecs_dom_b:
    line = plt.axvline(f, color='black', linestyle='--', linewidth=0.8)
    handles.append(line)
    labels.append(f'{f:.2f} Hz')
plt.legend(handles, labels, loc='upper right', fontsize=Fs+1, title_fontsize=Fs+1)
plt.xlabel('f[Hz]', fontsize=Fs+2)
plt.tick_params(axis='both', labelsize=Fs+2)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'{nombre}_fft_azul.png'))
plt.close()

# --- Lissajous ---
plt.figure(figsize=(6, 6))
sc = plt.scatter(x_rojo_centrado, x_azul_centrado, c=t_full, cmap='inferno', s=10)
cbar = plt.colorbar(sc)
cbar.set_label('t[s]', fontsize=Fs+1)
cbar.ax.tick_params(labelsize=Fs)
plt.xlabel('x1 [cm]', fontsize=Fs+1)
plt.ylabel('x2 [cm]', fontsize=Fs+1)
plt.tick_params(axis='both', labelsize=Fs)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, f'{nombre}_lissajous.png'))
plt.close()