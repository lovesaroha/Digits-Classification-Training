# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model on MNIST dataset.
from tensorflow import keras
import matplotlib.pyplot as plt

# Parameters.
learningRate = 0.01
epochs = 10
batchSize = 32

# Load MNIST dataset.
mnist = keras.datasets.mnist
(training_images, training_labels), (validation_images,
                                     validation_labels) = mnist.load_data()

# Normalize inputs.
training_images = training_images / 255
validation_images = validation_images / 255

# Create model with 10 output units for (0 to 9) digits.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Set loss function and optimizer.
model.compile(optimizer=keras.optimizers.Adam(learningRate),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
history = model.fit(training_images, training_labels, batch_size=batchSize, epochs=epochs, callbacks=[
                    checkAccuracy], validation_data=(validation_images, validation_labels))

# Predict on a random image.
image = validation_images[6]
prediction = model.predict(image.reshape(1, 28, 28))

# Show image.
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

# Show prediction.
print(prediction)
