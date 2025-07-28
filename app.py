from flask import Flask, request, jsonify, render_template
import tensorflow as tf

# 1. Initialize Flask
app = Flask(__name__)

# 2. Load trained model
MODEL_PATH = "animal_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 3. 
class_names = ['cat', 'dog']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

import numpy as np
from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error='No image provided'), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
    except:
        return jsonify(error='Invalid image'), 400

    # Preprocess: resize & normalize
    img = img.resize((256, 256))
    arr = np.array(img, dtype='float32') / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1,128,128,3)

    # Find result
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx]
    confidence = float(preds[idx])

    return jsonify(label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
