# -*- coding: utf-8 -*-
import tensorflow as tf  
import numpy as np  
"""
Created on Mon May  8 17:29:36 2017

@author: lankuohsing
"""


  
x=tf.Variable(tf.random_normal([7*7*64, 1024]))
x_shape=x.get_shape().as_list()[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(x_shape)
