import numpy as np
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants

# Load and preprocess MNIST Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Save the model as a SavedModel
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28], dtype=tf.float32)])
def serving_fn(x):
    return model(x)

# Specify a directory to save
model.save("saved_model_directory", save_format='tf')

# Convert the model to a frozen inference graph
loaded_model = tf.saved_model.load("saved_model_directory")
full_model = loaded_model.signatures['serving_default']

# Convert the model to constants
frozen_func = convert_to_constants.convert_variables_to_constants_v2(full_model)

# Save the frozen graph
tf.io.write_graph(frozen_func.graph, ".", "frozen_inference_graph.pb", as_text=False)
