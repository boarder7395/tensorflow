import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from mnist import MNIST
import os

root_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(root_dir, 'data')
model_dir = os.path.join(root_dir, 'models')

seed = 128
rng = np.random.RandomState(seed)

mndata = MNIST(data_dir)

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# define constants
time_steps = 28
num_units = 128
n_inputs = 28
learning_rate = 0.01
n_classes = 10
batch_size = 128

# weights and biases of appropriate shape to accomplish task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# defining placeholders
# image placeholder
x = tf.placeholder('float', [None, time_steps, n_inputs])

# input label placeholder
y = tf.placeholder('float', [None, ])

# define one hot encoder
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)

# processing the input tensor from [batch_size, n_steps, n_input] to "time_steps"
input = tf.unstack(x, time_steps, 1)

# defining the network
lstm_layer = rnn.BasicLSTMCell(num_units=num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype='float32')

# converting last output of dinmension [batch_size, num_units] to [batch_size, n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=onehot_labels))

# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    while epoch < 1000:

        for i in range(int(len(X_train)/batch_size)):
            batch_x = X_train[(i)*batch_size:(i+1)*batch_size, :]
            batch_y = y_train[(i)*batch_size:(i+1)*batch_size]

            batch_x = batch_x.reshape((batch_size, time_steps, n_inputs))

            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if epoch % 5 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print('For Epoch: {}'.format(epoch))
            print('Accuracy: {}'.format(acc))
            print('Loss: {}'.format(los))
            print('________________')

        epoch += 1

    # calculating test accuracy
    print('Testing Accuracy: {}'.format(sess.run(accuracy, feed_dict={x: X_test.reshape(-1, 28, 28), y: y_test})))
