from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "animal_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
 
class_names = ['cat', 'dog']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

import numpy as np

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error='No image provided'), 400

    file = request.files['image']
    
    try:
        # Read raw bytes
        image_bytes = file.stream.read()
        
        # Add size validation before processing
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            return jsonify(error='Image too large'), 400
        
        # Use TensorFlow's image pipeline (matches training)
        img = tf.io.decode_image(image_bytes, channels=3, dtype=tf.uint8)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32) / 255.0
        arr = tf.expand_dims(img, 0)  # Add batch dimension
        
    except Exception as e:
        return jsonify(error=f'Invalid image: {str(e)}'), 400

    # Find result
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx]
    confidence = float(preds[idx])

    return jsonify(label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
