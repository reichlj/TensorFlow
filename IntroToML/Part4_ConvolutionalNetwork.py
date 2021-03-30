import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
# training_images (60000,28,28)
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 64)        640
# max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0
# conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
# flatten (Flatten)            (None, 1600)              0
# dense (Dense)                (None, 128)               204928
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 243,786   Trainable params: 243,786  Non-trainable params: 0
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

print(test_labels[:100])
f, axarr = plt.subplots(3,5)
# Shoes : 0 23 28     7 26 Shirts
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,5):
    if x <4:
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0,x].grid(False)
        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1,x].grid(False)
        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2,x].grid(False)
    else:
        axarr[0, x].imshow(test_images[FIRST_IMAGE], cmap='inferno')
        axarr[0, x].grid(False)
        axarr[1, x].imshow(test_images[SECOND_IMAGE], cmap='inferno')
        axarr[1, x].grid(False)
        axarr[2, x].imshow(test_images[THIRD_IMAGE], cmap='inferno')
        axarr[2, x].grid(False)

plt.show()