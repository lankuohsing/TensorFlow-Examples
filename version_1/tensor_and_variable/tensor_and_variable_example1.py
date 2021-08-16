# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 23:39:42 2021

@author: lankuohsing
"""
import tensorflow as tf

a = tf.Variable(1.0,name='a')
b = tf.Variable(2.0,name='b')
c = tf.add(a,b)
d=tf.add(a,c)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
    # print(sess.run(d))
