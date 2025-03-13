import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

import shutil
from PIL import Image
from PIL.ExifTags import TAGS
import tqdm

def organize_images_by_header_labels(input_dir, output_dir, label_tag='ImageDescription'):
    """
    Organizes images into subdirectories based on metadata in their headers.

    Parameters:
        input_dir: Directory containing the images
        output_dir: Directory to create the organized structure
        label_tag: The EXIF tag to use as the label (default: 'ImageDescription')
    """

    # Track categories for statistics
    categories = {}
    unknown_count = 0

    # Process all images
    image_exts = ['.png']
    for filename in tqdm.tqdm(os.listdir(input_dir)):
        src_path = os.path.join(input_dir, filename)

        # Extract label from image header
        try:
            with Image.open(src_path) as img:
                exif_data = img._getexif()
                if exif_data is not None:
                    # Convert numerical EXIF tags to readable names
                    exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}

                    # Get label from specified tag
                    label = exif.get(label_tag)

                    if not label:
                        # Try some other common tags if the specified one is not found
                        for alt_tag in ['UserComment', 'XPComment', 'Comment']:
                            label = exif.get(alt_tag)
                            if label:
                                break
                else:
                    label = None
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            label = None

        # Clean up and process the label
        if label:
            if isinstance(label, bytes):
                label = label.decode('utf-8', errors='ignore').strip()
            label = label.strip().replace('/', '_')  # Sanitize for file path
        else:
            label = "unknown"
            unknown_count += 1

        # Create category directory if needed
        category_dir = os.path.join(output_dir, label)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        # Copy the image to its category directory
        dst_path = os.path.join(category_dir, filename)
        shutil.copy2(src_path, dst_path)

        # Update statistics
        categories[label] = categories.get(label, 0) + 1

# Usage example
# organized_dir = organize_images_by_header_labels('path/to/images', 'path/to/organized')

# Now create your Keras dataset
# from tensorflow.keras.utils import image_dataset_from_directory

# dataset = image_dataset_from_directory(
#     organized_dir,
#     image_size=(224, 224),
#     batch_size=32
# )

from datasets import load_dataset

image_size = (700, 700)
batch_size = 128
data_dir = "Eddie-train"
print(f"Checking directory existence: {os.path.exists(data_dir)}")
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "Eddie-train",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

        data_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


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
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
#keras.utils.plot_model(model, show_shapes=True)



epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)



img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")