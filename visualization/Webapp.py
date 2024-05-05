import numpy as np
import pandas as pd
from PIL import Image
import os
from tensorflow.keras.applications import VGG16, ResNet50
import requests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def extract_info_from_url(url):
    parts = url.split('/')
    year = parts[parts.index('photos') + 3]
    season = parts[parts.index('photos') + 4]
    type = parts[parts.index('photos') + 5]
    gender_age = parts[parts.index('photos') + 6]
    return type #year, season, type, gender_age


def load_images_from_urls(url_list, output_folder, files_in_folder):

    type_garments = {}  # Diccionario para almacenar imágenes por tipo de prenda
    for url, filename in zip(url_list, files_in_folder):
        if pd.notnull(url):  # Verifica si la URL no es nula
            try:
                img = Image.open(output_folder+filename)
                #img = img.resize((224, 224))  # Redimensionar todas las imágenes al mismo tamaño
                #img = np.array(img)
                garment_type = extract_info_from_url(url)  # Obtener el tipo de prenda desde la URL
                if garment_type not in type_garments:
                    type_garments[garment_type] = []
                type_garments[garment_type].append(img)  # Agregar la imagen al tipo de prenda correspondiente
            except Exception as e:
                pass
                #print(f"Error al procesar la URL {url}: {e}")
    return type_garments


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
        #if img.shape == (224, 224, 3):
        processed_images.append(img)
    return (processed_images)

# Función para extraer características de las imágenes
def extract_features(images):
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Añadir una dimensión para crear un lote de tamaño 1
        feature = base_model.predict(img)
        features.append(feature)
    return np.array(features)

def extract_features_resnet(images):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Añadir una dimensión para crear un lote de tamaño 1
        feature = base_model.predict(img)
        features.append(feature)
    return np.array(features)

def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

def select_n_components(features, explained_variance_threshold=0.95):
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_ratio_
    
    # Calcula la suma acumulativa de la varianza explicada
    cumulative_explained_variance = np.cumsum(explained_variance)
    
    # Encuentra el número de componentes que explican una cantidad específica de varianza
    n_components = np.argmax(cumulative_explained_variance >= explained_variance_threshold) + 1
    
    # Grafica la varianza explicada acumulativa
    plt.plot(np.arange(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.show()
    
    return n_components

def visualize_cluster(cluster_labels, images, cluster_number):
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_number]
    
    # Mostrar las imágenes del cluster
    num_images = len(cluster_indices)
    
    for i, idx in enumerate(cluster_indices):
        plt.figure(figsize=(5, 5))  # Tamaño de la figura
        plt.imshow(images[idx])  # Mostrar la imagen
        plt.axis('off')  # Ocultar los ejes
        plt.title(f'Image {idx}')  # Título con el índice de la imagen
        plt.show()  # Mostrar la imagen

def plot_similarity_matrix(cosine_similarities):
    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_similarities, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.show()
    
def get_similar_images(cosine_similarities, reference_image_index, urls, n=5):
    similarities = cosine_similarities[reference_image_index]
    similarities[reference_image_index] = -1
    top_indices = np.argsort(similarities)[-n:][::-1]
    print(urls[top_indices])
    return urls[top_indices]
    # fig, axes = plt.subplots(1, n+1, figsize=(15, 5))
    
    # axes[0].imshow(images[reference_image_index])
    # axes[0].set_title('Reference Image')
    # axes[0].axis('off')

    # for i, index in enumerate(top_indices, start=1):
    #     axes[i].imshow(images[index])
    #     axes[i].set_title(f'Similar Image {i}')
    #     axes[i].axis('off')
    # plt.show()
    

def main():
    data = pd.read_csv("../garments_dataset.csv")
    url_columns = ['IMAGE_VERSION_1', 'IMAGE_VERSION_2', 'IMAGE_VERSION_3']
    urls = data[url_columns].values
    input_folder = "/Users/manvirkaur/Desktop/jacat-n24/downloaded_images/"
    files_in_folder = os.listdir(input_folder)

    images = []
    urls = urls.flatten()
    type_garments = {}
    #images.extend(load_images_from_urls(url_row, output_folder))
    type_garments = load_images_from_urls(urls, input_folder, files_in_folder)

    #images = load_images_from_folder(input_folder)
    images_dict = type_garments['0']
    processed_images = preprocess_images(images_dict)
    #features = extract_features(processed_cpy)
    i = 0
    for j, img in enumerate(processed_images):
        if img.shape != (224, 224, 3):
            print("Index:", j)
            i = j

    #code before to find the index that didnt have shape (224, 224, 3) to delete it
    processed_cpy = processed_images.copy()
    features_resnet = extract_features_resnet(processed_cpy)
    flattened_features_resnet = features_resnet.reshape(len(features_resnet), 2048)

    n_components = select_n_components(flattened_features_resnet)
    reduced_features_pca = apply_pca(flattened_features_resnet, n_components=n_components)

    # KMeans
    num_clusters = 30 
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(reduced_features_pca)
    labels = kmeans.predict(reduced_features_pca)
    plt.scatter(reduced_features_pca[:, 0], reduced_features_pca[:, 1], c=labels, cmap='viridis')
    plt.title('Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # NearestNeighbors
    k = 5 
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(reduced_features_pca)
    distances, indices = knn.kneighbors(reduced_features_pca)
    plt.scatter(reduced_features_pca[:, 0], reduced_features_pca[:, 1], c=indices[:, 0], cmap='viridis')
    plt.title('Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    images_cpy = images_dict.copy()
    del images_cpy[i]

    visualize_cluster(labels, images_cpy, cluster_number=0)

    cosine_similarities = cosine_similarity(reduced_features_pca, reduced_features_pca)
    plot_similarity_matrix(cosine_similarities)

    reference_image_index = 1
    similarities = cosine_similarities[reference_image_index]
    top_indices = np.argsort(similarities)[-5:][::-1]
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    axes[0].imshow(images_cpy[reference_image_index])
    axes[0].set_title('Reference Image')
    axes[0].axis('off')

    for i, index in enumerate(top_indices, start=1):
        axes[i].imshow(images_cpy[index])
        axes[i].axis('off')
    plt.show()
    cosine_similarity_file = "./cosine_similarity.npy"
    np.save(cosine_similarity_file, cosine_similarities)
    return cosine_similarities, urls

#get_similar_images(cosine_similarities, index, urls, n=5)
cosine_similarities, urls = main()

        
        