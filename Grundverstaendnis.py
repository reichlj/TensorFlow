# https://data-science-blog.com/blog/2018/09/07/einfuhrung-in-tensorflow/
# https://data-science-blog.com/blog/2018/10/09/ii-einfuhrung-in-tensorflow-grundverstandnis-fur-tensorflow/
# requires Tensorflow 1.x
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Tutorial 1
print('Tensorflow-Version:', tf.__version__)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# Tutorial 2
# Eingangssignal, Angstzustand der Gäste -> je größer desto größer der Angstzustand
x_input = [[-10], [-5], [-2], [-1], [2], [1], [6], [9]]

# gewünschtes Ausgangssignal, Endzustand der Gäste: 1 bedeutet Wunsch nie gefahren zu sein
y_input = [[0], [0], [0], [0], [0], [0], [1], [1]]

# Platzhalter - Wagon, Eingangsgröße
wag = tf.placeholder(tf.float32, shape=[8, 1])
# gewünschter Endzustand der Gäste
y_true = tf.placeholder(tf.float32, shape=[8, 1])

# Variablen
v = tf.Variable([[1.0, ]])     # Geschwindigkeit des Wagons
h = tf.Variable([[-2.0]])      # Starthöhe des Wagons

# Knoten mit Matrizenoperator, Fahrelement, z.B. Airtime-Hügel
z = tf.matmul(wag, v) + h
# Knoten mit ReLu-Aktivierungsfunktion
y_pred = tf.nn.relu(z)
# Fehlerfunktion
err = tf.square(y_true - y_pred)
# Optimierer
opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(err)
# Initialisierung der Variablen
init = tf.global_variables_initializer()

runden = 100
# Array zum Speichern der Zwischenwerte
v_array = []
h_array = []
loss = []

# Ausführung des Graphen
with tf.Session() as sess:
    # Initialisierung dar Variablen
    sess.run(init)
    for i in range(runden):
        _, geschw, hoehe, Y_pred, error = \
            sess.run([opt, v, h, y_pred, err],
                     feed_dict={ wag: x_input, y_true: y_input } )
        loss.append(np.mean(error))
        v_array.append(float(geschw))
        h_array.append(float(hoehe))

print('Angstlvl: {}'.format(Y_pred))
print('Geschwindigkeit: {}'.format(geschw))
print('Starthöhe: {}'.format(hoehe))
print('Fehler: {}'.format(error))

fig, ax = plt.subplots(1,2, figsize=(15, 7))
ax[0].plot(range(runden), loss)
ax[0].set_xlabel('Runden/Epochen')
ax[0].set_ylabel('$err$')
ax[0].title.set_text('Fehlerverlauf')
ax[1].plot(range(runden), v_array, label='Geschwindigkeit')
ax[1].plot(range(runden), h_array, label='Bias')
ax[1].set_xlabel('Runden/Epochen')
ax[1].set_ylabel('$v, h$')
ax[1].legend()
ax[1].title.set_text('Geschwindigkeits- und Höhenverlauf')
plt.show()