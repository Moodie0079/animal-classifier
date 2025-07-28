from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model (in this case we are loading the lite model due to space issue with free Render hosting)
TFLITE_PATH = "animal_classifier_dr.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        img = Image.open(file.stream).convert('RGB')
    except:
        return jsonify(error='Invalid image'), 400

    # Preprocess: resize & normalize FOR h5 MODEL********
    # img = img.resize((256, 256))
    # arr = np.array(img, dtype='float32') / 255.0
    # arr = np.expand_dims(arr, axis=0)  # shape (1,128,128,3)

    # FOR UINT8 / INT8 MODEL*******  (keep input the model expects)
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 256, 256, 3)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], arr)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (it will be float32)
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    # Apply softmax to get probabilities
    preds = tf.nn.softmax(preds).numpy()

    idx   = int(np.argmax(preds))
    label = class_names[idx]
    confidence = float(preds[idx])

    return jsonify(label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
