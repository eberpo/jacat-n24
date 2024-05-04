import numpy as np
import pandas as pd
import requests
from PIL import Image
import wget
import os


# Función para cargar y preprocesar imágenes desde URLs
def load_images_from_urls(url_list, output_folder):
    images = []
    for url in url_list:
        if pd.notnull(url):  # Verifica si la URL no es nula
            try:
                filename = wget.download(url, out=output_folder)
                img = Image.open(filename)
                img = img.resize((224, 224))  # Redimensionar todas las imágenes al mismo tamaño
                img = np.array(img)
                images.append(img)
            except Exception as e:
                print(f"Error al procesar la URL {url}: {e}")
    return np.array(images)

# Cargar el dataset
data = pd.read_csv("garments_dataset.csv")

# Obtener las URLs de las imágenes
url_columns = ['IMAGE_VERSION_1', 'IMAGE_VERSION_2', 'IMAGE_VERSION_3']
urls = data[url_columns].values

output_folder = "downloaded_images"
os.makedirs(output_folder, exist_ok=True)

# Descargar y cargar las imágenes
images = []
for url_row in urls:
    images.extend(load_images_from_urls(url_row, output_folder))