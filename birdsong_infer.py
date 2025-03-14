import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pandas as pd
import numpy as np
import keras
from tensorflow import data as tf_data
import tensorflow as tf

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0  # Normalize to [0,1]
    return image, label

model = keras.models.load_model("save_at_25.keras")

image_size = (224, 224)  # Example size, you can experiment with different sizes
batch_size = 128
data_directory = './images/'
test_labels_df = pd.read_csv(data_directory + 'test_labels.csv')
test_image_paths = [data_directory + 'test/' + fname for fname in test_labels_df['filename']]
test_labels = test_labels_df['label'].values

train_labels_df = pd.read_csv(data_directory + 'train_labels.csv')
train_image_paths = [data_directory + 'train/' + fname for fname in train_labels_df['filename']]
train_labels = train_labels_df['label'].values


test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf_data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE)

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_ds = train_ds.map(load_image, num_parallel_calls=tf_data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE)

# Calculate class weights inversely proportional to frequency
class_weights = {}
for i, count in zip(unique, counts):
    class_weights[i] = len(train_labels) / (len(unique) * count)

# Apply in training
model.fit(train_ds, epochs=epochs, class_weight=class_weights)


print("Making predictions...")
predictions = model.predict(train_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in train_ds], axis=0)
accuracy = np.mean(predicted_classes == true_classes)

print("Number of unique predictions:", len(np.unique(predicted_classes)))
print("Most common predictions:", np.bincount(predicted_classes).argsort()[-2:])

print(f"Average accuracy: {accuracy:.2f}")