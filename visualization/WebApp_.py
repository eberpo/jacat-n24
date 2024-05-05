
import streamlit as st
from PIL import Image
import os
import numpy as np


def display_similar_images(paths):
    col1, col2, col3, col4, col5 = st.columns(5)
    image1 = Image.open(paths[0])
    image2 = Image.open(paths[1])
    image3 = Image.open(paths[2])
    image4 = Image.open(paths[3])
    image5 = Image.open(paths[4])
    col1.image(image1, use_column_width=True)
    col2.image(image2, use_column_width=True)
    col3.image(image3, use_column_width=True)
    col4.image(image4, use_column_width=True)
    col5.image(image5, use_column_width=True)

def get_similar_images(cosine_similarities, reference_image_index, urls, n=5):
    similarities = cosine_similarities[reference_image_index]
    similarities[reference_image_index] = -1
    top_indices = np.argsort(similarities)[-n:][::-1]
    print(urls[top_indices])
    return urls[top_indices]

# Get the list of image files from the images folder
images_path = "/Users/manvirkaur/Desktop/jacat-n24/downloaded_images"
image_files = os.listdir(images_path)
cosine_similarity_file = "./cosine_similarity.npy"
cosine_similarities = os.open(cosine_similarity_file)



# Create three columns in the web app
col1, col2, col3 = st.columns(3)

# Iterate over the image files and display them in the columns
for i, image_file in enumerate(image_files):
    # Load the image
    image = Image.open(os.path.join(images_path, image_file))

    # Display the image in the corresponding column
    if i % 3 == 0:
        col1.image(image, use_column_width=True)
        if col1.button(f"Button {i}"):
            # Get the similar images
            similar_images_urls = get_similar_images(cosine_similarities, i, image_files, 5)
            # Display the similar images
            display_similar_images(similar_images_urls)
    elif i % 3 == 1:
        col2.image(image, use_column_width=True)
        if col2.button(f"Button {i}"):
            similar_images_urls = get_similar_images(cosine_similarities, i, image_files, 5)
            display_similar_images(similar_images_urls)
    else:
        col3.image(image, use_column_width=True)
        if col3.button(f"Button {i}"):
            similar_images_urls = get_similar_images(cosine_similarities, i, image_files, 5)
            display_similar_images(similar_images_urls)
        