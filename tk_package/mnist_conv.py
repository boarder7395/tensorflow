from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
from mnist import MNIST

root_dir = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
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

split_size = int(len(X_train)*0.7)
train_x, val_x = X_train[:split_size,:], X_train[split_size:,:]
train_y, val_y = y_train[:split_size], y_train[split_size:]

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
    # define input layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # define the first convolutional layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # define the first pooling layer
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # define the second convolutional layer
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )

    # define the second pooling layer
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # add a third conv layer
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=1
    )

    # define the dense layer
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])
    dense = tf.layers.dense(
        inputs=pool3_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # define the logits layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    # define the predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # define one hot encoding
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    # calculate loss (for both TRAIN and EVAL)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def main(unused_argv):
    # Load training and eval data
    train_data = np.asarray(train_x, dtype=np.float32)
    train_labels = train_y
    eval_data = val_x
    eval_labels = val_y

    #     tensors_to_log = {"probabilities": "softmax_tensor"}
    #     logging_hook = tf.train.LoggingTensorHook(
    #       tensors=tensors_to_log, every_n_iter=50)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir + "/mnist_convnet_model4")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000000)
    #         hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.asarray(eval_data, dtype=np.float32)},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()