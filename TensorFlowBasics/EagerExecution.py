import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cProfile

print('TensorFlow-Version:', tf.__version__)
print('Eager-Execution', tf.executing_eagerly())

# ********* Setup and basic usage

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

# Enabling eager execution changes how TensorFlow operations behave
# now they immediately evaluate and return their values to Python.
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# Broadcasting support
b = tf.add(a, 1)
print(b)
# Operator overloading is supported
print(a * b)
# Use NumPy values
c = np.multiply(a, b)
print(c)
# Obtain numpy value from a tensor:
print(a.numpy())

# ********* Dynamic control flow

def fizzbuzz(max_num):
  # conditionals that depend on tensor values and it prints
  # these values at runtime
  counter = tf.constant(0)
  max_numTF = tf.convert_to_tensor(max_num)
  for num in range(1, max_numTF.numpy()+1):
    numTF = tf.constant(num)
    if int(numTF % 3) == 0 and int(numTF % 5) == 0:
      print('FizzBuzz')
    elif int(numTF % 3) == 0:
      print('Fizz')
    elif int(numTF % 5) == 0:
      print('Buzz')
    else:
      print(numTF.numpy())
    counter += 1

fizzbuzz(8)

# ********* Computing gradients
# During eager execution, use tf.GradientTape to trace operations for computing
# gradients later
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
print('Gradient', grad.numpy())  #=> [[2.]]

# ********* Train a model
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])

for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

def train_step(images, labels):
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)

    # Add asserts to check the shape of the output.
    tf.debugging.assert_equal(logits.shape, (32, 10))

    loss_value = loss_object(labels, logits)

  loss_history.append(loss_value.numpy().mean())
  grads = tape.gradient(loss_value, mnist_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train(epochs):
  for epoch in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset):
      train_step(images, labels)
    print ('Epoch {} finished'.format(epoch))

train(epochs = 3)

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

