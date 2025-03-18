import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import keras
from tensorflow import data as tf_data
import tensorflow as tf

model = keras.models.load_model("callback.keras")

image_size = (256, 256)
batch_size = 32
data_dir = './eddie_subdirs/'

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    seed=15506)

# After training with a few samples
predictions = model.predict(test_ds)
print("Prediction shape:", predictions.shape)
print("Sample raw outputs:")
print(predictions)
print("\nSum of each prediction row (should be close to 1.0 with softmax):")
print(np.sum(predictions, axis=1))
print("\nMax value in each prediction:")
print(np.max(predictions, axis=1))
print("\nPredicted classes:", np.argmax(predictions, axis=1))

print("Making predictions...")
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in test_ds], axis=0)
accuracy = np.mean(predicted_classes == true_classes)

print("Number of unique predictions:", len(np.unique(predicted_classes)))
print("Most common predictions:", np.bincount(predicted_classes).argsort()[-2:])

print(f"Average accuracy: {accuracy:.2f}")