import tensorflow as tf

c = tf.constant([[4.0, 5.0],[10.0, 1.0]])
sc = tf.nn.softmax(c)
print(sc)
print(tf.reduce_sum(sc))

c3 = tf.constant([[4.0, 5.0, 6.0],[10.0, 1.0, 12.0]])
print('argmax von 2*3-matrix',tf.argmax(c3))

v = tf.constant([2.0, 3.0, 4.0, 5.0])
print(v[1:])
print(v[1:-1])