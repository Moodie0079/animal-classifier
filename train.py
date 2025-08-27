import tensorflow as tf
import os

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping 

# Configures GPU memory growth, stops from using all available memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Preprocessing layers
normalization_layer = layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Dataset parameters
DATA_DIR    = "filtered_dataset"
IMG_SIZE    = (256, 256)   
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
SEED        = 123

# Load training set
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Capture class names now, before mapping
class_names = train_ds.class_names

# Load validation set
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Apply data augmentation + normalization to training data
train_ds = train_ds.map(
    lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Apply normalization only to validation data
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Classes: {class_names}")

# Model
model = models.Sequential([
    # 1st conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),

    # 2nd conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3rd conv block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten feature maps to a vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),  # Temporarily disabled

    # Output layer: 2 classes
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# EarlyStopping callback to stop training when val_loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model 
epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop]
)

# Save the model
model.save("animal_classifier.h5")
print("Model saved to animal_classifier.h5")