import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2

dy_dx = tape.gradient(y,x)
print('dy_dx',dy_dx.numpy())

w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

# using Arrays
with tf.GradientTape(persistent=True) as tape:
  y = x @ w + b
  loss = tf.reduce_mean(y**2)

der = tape.gradient(loss, [w, b])
[dl_dw, dl_db] = tape.gradient(loss, [w, b])

print(w.shape)
print(dl_dw.shape)
print(dl_dw)
print(dl_db)

# using Dictionarys
my_vars = {
    'w': w,
    'b': b
}

grad = tape.gradient(loss, my_vars)
grad['b']
print(grad)

# Gradients with respect to Model
# output = activation(dot(input, kernel) + bias) - features as column
layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant([[1., 2., 3.]])
# x=(1,3) one sample and 3 features - features as columns W=(3,2) X*W=(1,2)
# x=(4,3) four samples and 3 features - features as columns W=(3,2) X*W=(4,2)
#      W = (nodes in layer n-1, nodes in layer n)
# each row of X is a sample
# Ng x=(3,1) one sample and 3 features - features as rows W=(2,3) W*X=(2,1)
# Ng x=(3,4) four samples and 3 features - features as rows W=(2,3) W*X=(2,4)
# Ng each column of X is a sample
#      W = (nodes in layer n, nodes in layer n-1,)
#x = tf.constant([[1.], [2.], [3.]])

with tf.GradientTape() as tape:
  # Forward pass
  y = layer(x)
  loss = tf.reduce_mean(y**2)

# Calculate gradients with respect to every trainable variable
# trainable variables are W and b of Layer Dense(2)
# W is a 3*2-matrix and and b is a vector with 2 elements
grad = tape.gradient(loss, layer.trainable_variables)
for var, g in zip(layer.trainable_variables, grad):
  print(f'{var.name}, shape: {g.shape}')