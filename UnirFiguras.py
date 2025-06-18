import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
archivo_csv = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/2ResortesMetalicos1ResortePlastico/trayectorias.csv'
nombre = '2ResortesMetalicos1ResortePlastico '
tamano_puntos = 15
epsilon = 0.4
t_min = 0
t_max = 5
Fs = 16
ylim_fft = (1e-4, 1)
# ------------------------------

# Leer y preparar datos
df = pd.read_csv(archivo_csv).dropna()
t_full = df['tiempo_s'].to_numpy()
x_rojo = df['x_rojo'].to_numpy().astype(float)
x_azul = df['x_azul'].to_numpy().astype(float)
x_rojo_c = x_rojo - np.mean(x_rojo)
x_azul_c = x_azul - np.mean(x_azul)

# Filtrar para trayectorias
mask = (t_full >= t_min) & (t_full <= t_max)
t_traj = t_full[mask]
x_rojo_traj = x_rojo_c[mask]
x_azul_traj = x_azul_c[mask]

# Carpeta de salida
carpeta_base = os.path.dirname(os.path.abspath(archivo_csv))
carpeta_unida = os.path.join(carpeta_base, 'FigurasUnidas')
os.makedirs(carpeta_unida, exist_ok=True)

# Función FFT
def obtener_frecuencias_dominantes(y, t, epsilon):
    N = len(t)
    dt = t[1] - t[0]
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, dt)
    amplitudes = np.abs(fft_y)
    valid = freqs > 0
    freqs, amplitudes = freqs[valid], amplitudes[valid]
    orden = np.argsort(amplitudes)[::-1]
    dom = []
    for idx in orden:
        f = freqs[idx]
        if all(abs(f - d) > epsilon for d in dom):
            dom.append(f)
        if len(dom) == 2:
            break
    return freqs, amplitudes, dom

# Crear figura 2x2
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Etiquetas (a)-(d)
letras = ['a)', 'b)', 'c)', 'd)']
for ax, letra in zip(axs.flat, letras):
    ax.text(0.02, 0.95, letra, transform=ax.transAxes, fontsize=Fs, fontweight='bold', va='top')

# Trayectoria roja (izquierda superior)
axs[0, 0].plot(t_traj, x_rojo_traj, color='red', alpha=0.6)
axs[0, 0].scatter(t_traj, x_rojo_traj, color='red', s=tamano_puntos)
axs[0, 0].set_xlabel('t [s]', fontsize=Fs)
axs[0, 0].set_ylabel('x1 [cm]', fontsize=Fs)
axs[0, 0].tick_params(labelsize=Fs-1)

# Trayectoria azul (derecha superior)
axs[0, 1].plot(t_traj, x_azul_traj, color='blue', alpha=0.6)
axs[0, 1].scatter(t_traj, x_azul_traj, color='blue', s=tamano_puntos)
axs[0, 1].set_xlabel('t [s]', fontsize=Fs)
axs[0, 1].set_ylabel('x2 [cm]', fontsize=Fs)
axs[0, 1].tick_params(labelsize=Fs-1)

# FFT rojo (izquierda inferior)
freqs_r, amp_r, dom_r = obtener_frecuencias_dominantes(x_rojo_c, t_full, epsilon)
amp_r_norm = amp_r / np.max(amp_r)
axs[1, 0].semilogy(freqs_r, amp_r_norm, color='red')
for f in dom_r:
    axs[1, 0].axvline(f, color='black', linestyle='--', linewidth=0.8)
axs[1, 0].set_xlabel('f [Hz]', fontsize=Fs)
axs[1, 0].set_ylabel('Amplitud (norm.)', fontsize=Fs)
axs[1, 0].tick_params(labelsize=Fs-1)
axs[1, 0].set_ylim(ylim_fft)
legend_lines_r = [
    Line2D([0], [0], color='red', label='Espectro'),
    Line2D([0], [0], color='black', linestyle='--', label=f'f₁ = {dom_r[0]:.2f} Hz'),
    Line2D([0], [0], color='black', linestyle='--', label=f'f₂ = {dom_r[1]:.2f} Hz')
]
axs[1, 0].legend(handles=legend_lines_r, fontsize=Fs-2)

# FFT azul (derecha inferior)
freqs_b, amp_b, dom_b = obtener_frecuencias_dominantes(x_azul_c, t_full, epsilon)
amp_b_norm = amp_b / np.max(amp_b)
axs[1, 1].semilogy(freqs_b, amp_b_norm, color='blue')
for f in dom_b:
    axs[1, 1].axvline(f, color='black', linestyle='--', linewidth=0.8)
axs[1, 1].set_xlabel('f [Hz]', fontsize=Fs)
axs[1, 1].set_ylabel('Amplitud (norm.)', fontsize=Fs)
axs[1, 1].tick_params(labelsize=Fs-1)
axs[1, 1].set_ylim(ylim_fft)
legend_lines_b = [
    Line2D([0], [0], color='blue', label='Espectro'),
    Line2D([0], [0], color='black', linestyle='--', label=f'f₁ = {dom_b[0]:.2f} Hz'),
    Line2D([0], [0], color='black', linestyle='--', label=f'f₂ = {dom_b[1]:.2f} Hz')
]
axs[1, 1].legend(handles=legend_lines_b, fontsize=Fs-2)

# Guardar figura
plt.tight_layout()
plt.savefig(os.path.join(carpeta_unida, f'{nombre}_figura_unida.png'), dpi=300)
plt.close()
