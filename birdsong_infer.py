import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import keras
from tensorflow import data as tf_data

model = keras.models.load_model("save_at_25.keras")

image_size = (224, 224)  # Example size, you can experiment with different sizes
batch_size = 128
data_directory = './images/'
test_labels_df = pd.read_csv(data_directory + 'test_labels.csv')
test_image_paths = [data_directory + 'test/' + fname for fname in test_labels_df['filename']]

test_labels = test_labels_df['label'].values
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf_data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE)

print("Making predictions...")
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in test_ds], axis=0)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Average accuracy: {accuracy:.2f}")