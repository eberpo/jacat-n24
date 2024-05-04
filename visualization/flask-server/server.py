from flask import Flask, jsonify, send_from_directory
import os
import flask

app = Flask(__name__, static_folder='imageTest')  # Set the folder where your images are stored

# Endpoint to serve specific images by filename
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.static_folder, filename)

# Endpoint to list all images in the folder
@app.route('/all-images')
def list_images():
    images = os.listdir(app.static_folder)
    response = flask.jsonify(images)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5001)
    

