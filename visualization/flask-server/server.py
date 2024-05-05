from flask import Flask, jsonify, send_from_directory, request
#from flask_cors import CORS
import os
from model_vgg16 import main, visualize_similar_images  # Adjust this import as necessary

app = Flask(__name__, static_folder='imageTest')
#CORS(app)  # Allow cross-origin requests if your frontend is on a different domain/port

# Global variables to store model data
cosine_similarities, images_cpy, urls = None, None, None

@app.route('/init', methods=['GET'])
def initialize_model():
    global cosine_similarities, images_cpy, urls
    cosine_similarities, images_cpy, urls = main()  # main function prepares and loads model and data
    return jsonify({"message": "Model initialized and data prepared."})

@app.route('/similar-images', methods=['POST'])
def get_similar_images():
    data = request.json
    index = data.get('index')
    if index is None or not isinstance(index, int):
        return jsonify({"error": "Index must be a valid integer"}), 400
    similar_images = visualize_similar_images(cosine_similarities, images_cpy, index, urls)
    return jsonify(similar_images)

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/all-images')
def list_images():
    images = os.listdir(app.static_folder)
    response = jsonify(images)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(debug=True, port=5001)
