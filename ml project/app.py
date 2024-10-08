from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model('signature_recognition_model.h5')
label_encoder = joblib.load('label_encoder.pkl')  # Save the label encoder separately

def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return img

def predict_signature(image_path, true_label):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    prediction = model.predict(img)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0] == true_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            true_label = request.form.get('true_label')
            is_genuine = predict_signature(file_path, true_label)
            return render_template('index.html', filename=filename, is_genuine=is_genuine)
    return render_template('index.html')

if _name_ == "_main_":
    app.run(debug=True)