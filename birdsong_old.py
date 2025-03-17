import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
import pandas as pd
import matplotlib.pyplot as plt

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0,1]
    return image, label


image_size = (244, 244)
batch_size = 128
data_dir = "images"

print("Loading labels...")
data_directory = './images/'

# Load labels
train_labels_df = pd.read_csv(data_directory + 'train_labels.csv')
test_labels_df = pd.read_csv(data_directory + 'test_labels.csv')

# Load images
print("Loading image paths and labels...")
train_image_paths = [data_directory + 'train/' + fname for fname in train_labels_df['filename']]
train_labels = train_labels_df['label'].values

test_image_paths = [data_directory + 'test/' + fname for fname in test_labels_df['filename']]
test_labels = test_labels_df['label'].values


train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE)

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,

    image_size=image_size,
    batch_size=batch_size,
)

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x), y))

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

num_classes = 264  # Update this to the actual number of classes in your dataset

print("Creating the model...")
model = make_model(input_shape=image_size + (3,), num_classes=264)

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

print("Compiling the model...")
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

print("Starting training...")
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
# plt.imshow(img)

# img_array = keras.utils.img_to_array(img)
# img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(train_ds)
score = float(keras.ops.sigmoid(predictions[0][0]))
predicted_classes = np.argmax(predictions, axis=1)
print("Number of unique predictions:", len(np.unique(predicted_classes)), "Score: ", score)
print("Most common predictions:", np.bincount(predicted_classes).argsort()[-5:])

true_classes = np.concatenate([y for x, y in train_ds], axis=0)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Average accuracy: {accuracy:.2f}")