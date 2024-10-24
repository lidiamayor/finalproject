import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance
import random

# Crear carpetas de taked_photos y new_photos
taked_photos_path = 'taked_photos'
new_photos_path = 'new_photos'
categories = ['focus', 'tired', 'distracted']
distribution = {'train': 0.65, 'val': 0.20, 'test': 0.15}

# Crear carpetas para taked_photos
for category in categories:
    os.makedirs(f'{taked_photos_path}/{category}', exist_ok=True)

# Crear carpetas para new_photos
for folder in distribution.keys():
    for category in categories:
        os.makedirs(f'{new_photos_path}/{folder}/{category}', exist_ok=True)

# Función para capturar imágenes y guardarlas
def capture_images():
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    current_category = 0  # Índice para iterar sobre las categorías
    print("Presiona 'f' para capturar la imagen. Presiona 'q' para salir.")
    
    while current_category < len(categories):
        category = categories[current_category]
        print(f'Capturando imágenes para {category}. Presiona "f" para capturar.')

        images_captured = 0  # Contador de imágenes capturadas por categoría
        captured_images = set()  # Conjunto para evitar imágenes duplicadas

        while images_captured < 2:  # Capturar 2 imágenes por categoría
            ret, frame = cap.read()
            if not ret:
                break

            # Mostrar la imagen en la ventana
            cv2.imshow('Capture', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f'{taked_photos_path}/{category}/{category}_{timestamp}.jpg'
                
                if filepath not in captured_images:  # Verificar duplicados
                    cv2.imwrite(filepath, frame)
                    print(f'Imagen guardada en: {filepath}')
                    captured_images.add(filepath)  # Añadir a conjunto
                    images_captured += 1  # Incrementar el contador de imágenes
                else:
                    print("Imagen duplicada, intenta nuevamente.")
            elif key == ord('q'):  # Permitir que el usuario salga en cualquier momento
                cap.release()
                cv2.destroyAllWindows()
                print("Captura cancelada.")
                return

        current_category += 1  # Pasar a la siguiente categoría

    cap.release()
    cv2.destroyAllWindows()
    print("Captura completada.")

# Función para aplicar aumentación de imágenes
def augment_image(image, save_dir, category, prefix):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 1. Rotación
    for angle in [15, -15, 30, -30]:
        rotated = img.rotate(angle)
        rotated.save(f'{save_dir}/{category}/{prefix}_rotated_{angle}.jpg')

    # 2. Inversión horizontal
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(f'{save_dir}/{category}/{prefix}_flipped.jpg')

    # 3. Brillo
    enhancer = ImageEnhance.Brightness(img)
    for factor in [0.5, 1.5]:  # Brillo reducido y aumentado
        brightened = enhancer.enhance(factor)
        brightened.save(f'{save_dir}/{category}/{prefix}_brightness_{factor}.jpg')

    # 4. Zoom (Recortar la imagen)
    width, height = img.size
    for crop_factor in [0.8, 0.6]:  # Zoom con diferentes factores de recorte
        left = (1 - crop_factor) * width / 2
        top = (1 - crop_factor) * height / 2
        right = (1 + crop_factor) * width / 2
        bottom = (1 + crop_factor) * height / 2
        cropped = img.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
        cropped.save(f'{save_dir}/{category}/{prefix}_zoom_{crop_factor}.jpg')

# Función para aumentar imágenes y repartirlas en new_photos
def augment_and_distribute_images():
    for category in categories:
        images = []  # Lista para almacenar las imágenes originales

        # Leer imágenes de la carpeta de taked_photos
        for img_name in os.listdir(f'{taked_photos_path}/{category}'):
            img_path = f'{taked_photos_path}/{category}/{img_name}'
            img = cv2.imread(img_path)
            images.append((img, img_name))

        # Guardar aumentaciones
        for img, img_name in images:
            # Generar y guardar aumentaciones
            prefix = img_name.split(".")[0]
            augment_image(img, taked_photos_path, category, prefix)

        # Recolectar todas las imágenes (originales + aumentadas)
        all_images = []
        for img_name in os.listdir(f'{taked_photos_path}/{category}'):
            img_path = f'{taked_photos_path}/{category}/{img_name}'
            img = cv2.imread(img_path)
            all_images.append((img, img_name))

        # Repartir imágenes en train, val y test
        random.shuffle(all_images)  # Mezclar las imágenes para una mejor distribución
        total_images = len(all_images)
        train_count = int(distribution['train'] * total_images)
        val_count = int(distribution['val'] * total_images)
        
        # Guardar en train
        for img, img_name in all_images[:train_count]:
            cv2.imwrite(f'{new_photos_path}/train/{category}/{img_name}', img)

        # Guardar en val
        for img, img_name in all_images[train_count:train_count + val_count]:
            cv2.imwrite(f'{new_photos_path}/val/{category}/{img_name}', img)

        # Guardar en test
        for img, img_name in all_images[train_count + val_count:]:
            cv2.imwrite(f'{new_photos_path}/test/{category}/{img_name}', img)

# Capturar imágenes para cada categoría
capture_images()

# Aumentar imágenes y repartir en new_photos
augment_and_distribute_images()

print("Proceso completado. Las imágenes se han guardado en 'taked_photos' y se han repartido en 'new_photos'.")
