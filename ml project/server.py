from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Dummy model for demonstration purposes
def dummy_model(image_array):
    # Here you would load and use your actual machine learning model
    return "Signature Recognized"  # Dummy response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image_array = np.array(image)

    # Get prediction from the model
    prediction = dummy_model(image_array)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
