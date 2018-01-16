# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:37:00 2018

@author: lankuohsing
"""
#from tensorflow.models.rnn.ptb import reader
import tensorflow as tf;
import numpy as np;

A = [[1, 2, 3], [4, 2, 3]]
B = tf.unstack(A, axis=1)

with tf.Session() as sess:
    print(sess.run(B))