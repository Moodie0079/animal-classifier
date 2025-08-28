from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

MODEL_PATH = "animal_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Warm-up prediction to compile graph/cache for faster first request
# _ = model.predict(np.zeros((1, 256, 256, 3), dtype=np.float32), verbose=0)

@app.route('/healthz')
def healthz():
    return "ok", 200

 
class_names = ['cat', 'dog']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify(error='No image provided'), 400

    file = request.files['image']
    
    try:
        # Read raw bytes once
        raw = file.read()
        
        # Robust TF-only decode with animation protection
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # no GIF stacks
        img.set_shape([None, None, 3])
        
        # Fast downscale to model size with antialiasing
        img = tf.image.resize(img, [256, 256], antialias=True)
        img = tf.cast(img, tf.float32) / 255.0
        arr = tf.expand_dims(img, 0).numpy()  # Convert to numpy for model.predict
        
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
