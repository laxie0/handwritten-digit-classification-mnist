import tensorflow as tf
import numpy as np

# Load Dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build the Neural Network Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)  # Output layer (logits)
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model Summary (Optional but Recommended)
model.summary()

# Train the Model
model.fit(x_train, y_train, epochs=5)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Create Probability Model
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Make Predictions
predictions = probability_model(x_test[:5])

print("\nPredicted Probabilities for first 5 test images:")
print(predictions.numpy())

print("\nPredicted Digits:")
print(np.argmax(predictions.numpy(), axis=1))

print("\nActual Digits:")
print(y_test[:5])

# Save the Model
model.save("mnist_digit_classifier.h5")
print("\nModel saved as mnist_digit_classifier.h5")
