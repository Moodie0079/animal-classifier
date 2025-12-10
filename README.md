# Image Classifier

This is web app classifies images of cats and dogs using a custom-trained neural network. Upload a photo of a cat or dog and the model will analyize it to predict which animal it is, returning a confidence score. The model was trained from scratch on about 19,000 labeled images.

**Note**: The model performs significantly better with close-up images where the animal's face is clearly visible. Images with the subject far away or partially obscured may produce less accurate results.

## How it works

**Model**: A convolutional neural network built with TensorFlow. Three convolution layers extract features from the image, then dense layers make the final classification.

**Training**: The model uses data augmentation (random flips, rotations, zooms) to avoid overfitting, an issue which I had experienced. Early stopping monitors validation loss and saves the best version.

**Backend**: Flask serves the web interface and handles predictions. Images are preprocessed to 256x256 pixels and fed through the model.

**Frontend**: Plain HTML with TailwindCSS. Shows image preview, prediction results, and a confidence bar.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open your browser to `http://localhost:5000`

## Training the model

If you want to retrain:

```bash
python train.py
```

This loads images from `filtered_dataset/`, trains the model, and saves it as `animal_classifier.h5`.

## Files

- `app.py` - Flask web server and prediction endpoint
- `train.py` - Model training script
- `animal_classifier.h5` - Trained model weights
- `templates/index.html` - Frontend interface
- `filtered_dataset/` - Training images (Cat and Dog folders)

## Tech stack

- Python
- TensorFlow
- Flask