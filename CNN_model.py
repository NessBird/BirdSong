import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

print("Loading labels...")
data_directory = '/content/drive/MyDrive/images/images/'

# Load labels
train_labels_df = pd.read_csv(data_directory + 'train_labels.csv')
test_labels_df = pd.read_csv(data_directory + 'test_labels.csv')

print("Loading image paths and labels...")
# Load images
train_image_paths = [data_directory + 'train/' + fname for fname in train_labels_df['filename']]
train_labels = train_labels_df['label'].values

test_image_paths = [data_directory + 'test/' + fname for fname in test_labels_df['filename']]
test_labels = test_labels_df['label'].values

# Set image size and batch size
image_size = (224, 224)  # EfficientNetB0 default input size
batch_size = 128


# Function to load and preprocess images
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

print("Creating TensorFlow datasets...")
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Defining data augmentation layers...")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

print("Defining EfficientNetB0 model...")
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False  # Freeze the base model
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

num_classes = 264
print("Creating the model...")
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)

print("Compiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)


print("Starting training...")
epochs = 2
callbacks = [keras.callbacks.ModelCheckpoint("efficientnet_model_{epoch}.keras")]
model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=callbacks)

# Load and preprocess an example image for prediction
print("Loading and preprocessing example image...")
img = keras.utils.load_img("/content/drive/MyDrive/images/images/train/spectrogram_10.png", target_size=image_size)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

print("Making predictions...")
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in test_ds], axis=0)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Average accuracy: {accuracy:.2f}")

print("Making predictions...")
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class[0]}")


# Evaluate the model on the test dataset
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")
