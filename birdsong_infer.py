import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import keras
from tensorflow import data as tf_data
import tensorflow as tf

model = keras.models.load_model("callback.keras")

img_height = 256
img_width = 256
batch_size = 32
data_dir = './eddie_subdirs/'

prediction_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    shuffle=False,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=15506)

# Check if the model can still predict from the dataset. It's really the same as the training set + validation set.
results = model.evaluate(prediction_ds)

print(f"Test loss: {results[0]:.2f}")
print(f"Test accuracy: {results[1]:.2f}")