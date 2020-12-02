import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(test_images[1].shape)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

network.save('./models')

test_lost, test_acc = network.evaluate(test_images, test_labels)

# network = models.load_model('./models', compile=True)

# print(test_images[:1].shape)
# print(type(test_images[:1]))

# prediction = network.predict(test_images[:1])
# print(prediction.shape)

print(f'test_acc: {test_acc}')