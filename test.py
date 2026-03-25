import cv2
import torch
import os
from ultralytics import YOLO
import numpy as np

RESIZE_WIDTH = 320
RESIZE_HEIGHT = 320

def to_binary(image, binary_threshold):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def keep_largest_component(binary):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    largest_label = 1
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    result = np.zeros_like(binary, dtype=np.uint8)
    result[labels == largest_label] = 255

    return result

model_path = "models/YOLO/best_large_model_yolo.pt"
model = YOLO(model_path)

data_dir = "data"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"El directorio {data_dir} no existe.")

data_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not data_files:
    raise FileNotFoundError(f"No se encontraron imágenes en {data_dir}.")

for image_name in data_files:
    image_path = os.path.join(data_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        continue

    image_resized = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    binary_image = to_binary(image_resized, binary_threshold=128)

    largest_component = keep_largest_component(binary_image)

    largest_component_rgb = cv2.cvtColor(largest_component, cv2.COLOR_GRAY2BGR)

    largest_component_rgb = cv2.bitwise_not(largest_component_rgb)

    results = model.predict(largest_component_rgb, verbose=True)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
        confidence = box.conf[0]  # Confianza
        class_id = int(box.cls[0])  # ID de la clase
        class_name = results.names[class_id]  # Nombre de la clase

        # Dibujar la caja y la etiqueta
        label = f"{class_name} ({confidence:.2f})"
        cv2.rectangle(largest_component_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caja verde
        cv2.putText(largest_component_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen procesada con las detecciones
    cv2.imshow(f"YOLO Detections - {image_name}", largest_component_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()