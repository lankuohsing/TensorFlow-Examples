# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:40:23 2021

@author: lankuohsing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# In[]
# Build your graph.
"""
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = 3*x+2
w = tf.Variable(tf.random_uniform([2, 1]))
y_pred = tf.math.add(tf.math.multiply(x,w[0]),w[1])
# ...
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
# init_op = w.initializer
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # `sess.graph` provides access to the graph used in a `tf.Session`.
    writer = tf.summary.FileWriter("./log/", sess.graph)

    # Perform your computation...
    for i in range(100):
        _,w_,y_,loss_=sess.run(train_op,w,y_pred,loss)
        print(w_,y_,loss_)

    writer.close()
"""
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log/", sess.graph)
    sess.run(init)

    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    print(sess.run(y_pred))
    writer.close()