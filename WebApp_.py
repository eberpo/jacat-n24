import streamlit as st
from PIL import Image
import os
import numpy as np
import pandas as pd


images_path = "/Users/manvirkaur/Desktop/jacat-n24/downloaded_images"
image_files = os.listdir(images_path)

cosine_similarity_file = "./cosine_similarity.csv"
cosine_df = pd.read_csv(cosine_similarity_file)
cosine_similarities = cosine_df.to_numpy()

urls_found_files = "./urls_found.csv"
urls_df = pd.read_csv(urls_found_files)
urls_found = urls_df.to_numpy()

# Initialize session state variables
if 'show_similar' not in st.session_state:
    st.session_state.show_similar = False
if 'similar_images' not in st.session_state:
    st.session_state.similar_images = []

def display_similar_images(paths):
    st.title("Similar Images")
    cols = st.columns(5)
    images = [Image.open(os.path.join(images_path, path)) for path in paths]
    for col, img in zip(cols, images):
        col.image(img, use_column_width=True)

def get_similar_images(cosine_similarities, reference_image_index, urls, n=5):
    similarities = cosine_similarities[reference_image_index]
    similarities[reference_image_index] = -1
    top_indices = np.argsort(similarities)[-n:][::-1]
    similar_urls = [urls[i] for i in top_indices]
    return similar_urls

def show_gallery():
    st.title("Image Gallery")
    cols = st.columns(3)
    for i, image_file in enumerate(image_files):
        image = Image.open(os.path.join(images_path, image_file))
        col = cols[i % 3]
        col.image(image, use_column_width=True)
        if col.button(f"Show Similar {i}", key=i):
            print(i)
            similar_images_urls = get_similar_images(cosine_similarities, i, image_files, 5)
            st.session_state.similar_images = similar_images_urls
            st.session_state.show_similar = True

# Main app logic
if st.session_state.show_similar:
    st.button("Back to Gallery", on_click=lambda: setattr(st.session_state, 'show_similar', False))
    display_similar_images(st.session_state.similar_images)
else:
    show_gallery()
