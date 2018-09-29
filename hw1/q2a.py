""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.01
batch_size = 128
n_epochs = 50

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)


X = tf.placeholder(tf.float32, [batch_size, 784], name="X_placeholder")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="Y_placeholder")

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name='bias')

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)

    for i in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: x_batch, Y: y_batch})
            total_loss += loss_batch
        print ('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')



    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for idx in range(n_batches):
        x_batch, y_batch = mnist.test.next_batch(batch_size)
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=0)
        accuracy_batch = sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))

    writer.close()

