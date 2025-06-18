import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el archivo CSV (reemplazá con tu ruta)
ruta_csv = '/Users/admin/Documents/Fisica experimental/Experimento-Modos-Normales-de-Oscilacion/MasasIgualesMetalicosModoLento/trayectorias.csv'
df = pd.read_csv(ruta_csv)

# Extraer los datos necesarios
t = df['tiempo_s'].values
x_rojo = df['x_rojo'].values
x_azul = df['x_azul'].values

# Figura de Lissajous coloreada por el tiempo
plt.figure(figsize=(6, 6))
sc = plt.scatter(x_rojo, x_azul, c=t, cmap='plasma', s=5)  # color por tiempo
plt.xlabel('Posición Carrito Rojo [px]')
plt.ylabel('Posición Carrito Azul [px]')
plt.title('Figura de Lissajous coloreada por el tiempo')
plt.grid(True)
plt.axis('equal')

# Agregar barra de color
cbar = plt.colorbar(sc)
cbar.set_label('Tiempo [s]')

plt.tight_layout()
plt.show()



# Simulación de datos
f = 0.5  # frecuencia en Hz
T = 10   # duración en segundos
N = 1000 # cantidad de puntos

t = np.linspace(0, T, N)
A = 50   # amplitud en píxeles

# Movimiento perfectamente en fase (modo lento)
x_azul = A * np.cos(2 * np.pi * f * t)
x_rojo = A * np.cos(2 * np.pi * f * t)

# Gráfico Lissajous coloreado por tiempo
plt.figure(figsize=(6, 6))
sc = plt.scatter(x_rojo, x_azul, c=t, cmap='plasma', s=5)
plt.xlabel('Posición Carrito Rojo [px]')
plt.ylabel('Posición Carrito Azul [px]')
plt.title('Figura de Lissajous (modo en fase simulado)')
plt.grid(True)
plt.axis('equal')

# Barra de color
cbar = plt.colorbar(sc)
cbar.set_label('Tiempo [s]')

plt.tight_layout()
plt.show()

