import cv2
import time

# --- Inicializar cámara (puede ser 0 o 1 dependiendo de tu puerto) ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# --- Configurar resolución opcional ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Medir FPS reales ---
num_frames = 100
print(f"Midiendo FPS reales con {num_frames} cuadros...")

start = time.time()
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
end = time.time()

# --- Calcular FPS ---
seconds = end - start
fps_real = num_frames / seconds

print(f"✅ FPS reales: {fps_real:.2f} fps")

# --- Liberar cámara ---
cap.release()