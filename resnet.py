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

# Block.


class Block(keras.Model):
    def __init__(self, filters, kernel_size):
        super(Block, self).__init__(name="")
        # Convolutional layer.
        self.convOne = keras.layers.Conv2D(
            filters, kernel_size, padding="same")
        # Batch normalize.
        self.batchNormalizeOne = keras.layers.BatchNormalization()

        # Convolutional layer.
        self.convTwo = keras.layers.Conv2D(
            filters, kernel_size, padding="same")
        # Batch normalize.
        self.batchNormalizeTwo = keras.layers.BatchNormalization()

        # Activation.
        self.activation = keras.layers.Activation("relu")
        self.add = keras.layers.Add()

    def call(self, input_tensor):
        x = self.convOne(input_tensor)
        x = self.batchNormalizeOne(x)
        x = self.activation(x)

        x = self.convTwo(x)
        x = self.batchNormalizeTwo(x)
        x = self.activation(x)

        x = self.add([x, input_tensor])
        x = self.activation(x)
        return x


# Residual Network.

class ResNet(keras.Model):
    def __init__(self, total_classes):
        super(ResNet, self).__init__()
        # Convolutional layer.
        self.conv = keras.layers.Conv2D(
            64, (7,7), padding="same")
        # Batch normalize.
        self.batchNormalize = keras.layers.BatchNormalization()
        # Activation.
        self.activation = keras.layers.Activation("relu")
        # Max pool.
        self.max_pool = keras.layers.MaxPool2D((3, 3))

        # Blocks.
        self.blockOne = Block(64, 3)
        self.blockTwo = Block(64, 3)

        # Global pool.
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        # Output layer.
        self.output_layer = keras.layers.Dense(total_classes, activation="softmax")

    def call(self, inputs):
      x = self.conv(inputs)
      x = self.batchNormalize(x)
      x = self.activation(x)
      x = self.max_pool(x)
      x = self.blockOne(x)
      x = self.blockTwo(x)
      x = self.global_pool(x)
      return self.output_layer(x)

# Load MNIST dataset.
mnist = keras.datasets.mnist
(training_images, training_labels), (validation_images,
                                     validation_labels) = mnist.load_data()

# Normalize inputs.
training_images = training_images / 255
validation_images = validation_images / 255

# Reshape input data for convolutional layer.
training_images = training_images.reshape(60000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)

# Create model with 10 output units for (0 to 9) digits.
model = ResNet(10)

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
history = model.fit(training_images, training_labels, epochs=epochs, callbacks=[
                    checkAccuracy], validation_data=(validation_images, validation_labels))

# Predict on a random image.
image = validation_images[6]
prediction = model.predict(image.reshape(1, 28, 28, 1))

# Show image.
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

# Show prediction.
print(prediction)
