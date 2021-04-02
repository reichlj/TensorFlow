# https://www.youtube.com/watch?v=bemDFpNooA8
import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9):
            print("\nReached 90% accuracy so cancelling training!")
            model.stop_training = True

fashion_mnist = keras.datasets.fashion_mnist
# training_images (60000,28,28), image=28x28, labels a number between 0 and 9
(train_images, train_labels), (test_images, test_labels) = \
                               fashion_mnist.load_data()
# normalize to [0,1] - better numerical behaviour
train_images = train_images/255
test_images = test_images/255

# Sequential : define a sequence of layers
# Flatten : reshape 28*28 to a vector
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax) ])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
#callbacks = myCallback()
#model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])

print('Validate using test_images')
test_loss = model.evaluate(test_images, test_labels)
print('TestLoss',test_loss)

predictions = model.predict(test_images[0:1,:,:])
print(predictions)