# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:46:52 2017

@author: lankuohsing
"""
import tensorflow as tf 
input1 = tf.constant([[1,2,3],[4,5,6]])  
with tf.Session() as sess:  
    print(sess.run(tf.shape(input1)))
