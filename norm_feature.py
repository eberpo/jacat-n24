import numpy as np
import pandas as pd
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Función para cargar imágenes desde una carpeta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))
            images.append(img)
        except Exception as e:
            print(f"Error al cargar la imagen {filename}: {e}")
    return images

# Función para redimensionar y normalizar imágenes
def preprocess_images(images, size=(224, 224)):
    processed_images = []
    for img in images:
        img = img.resize(size)
        img = np.array(img) / 255.0
        processed_images.append(img)
    return (processed_images)

# Función para extraer características de las imágenes
def extract_features(images):
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = base_model.predict(images)
    return features


# Carpeta donde están las imágenes descargadas
input_folder = "downloaded_images"

# Cargar las imágenes desde la carpeta
images = load_images_from_folder(input_folder)

processed_images = preprocess_images(images)
print(processed_images)

# Extraer características de las imágenes
#features = extract_features(processed_images)

# Imprimir las dimensiones de las características
#print("Dimensiones de las características:", features.shape)
