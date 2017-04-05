# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:02:20 2017
Create a Constant op
Create a session and close it after using it
@author: lankuohsing
"""
import tensorflow as tf
# build a graph
a = 'Hello World! Welcome to Tensorflow'
hello = tf.constant(a, dtype=tf.string)

# launch the graph in a session
sess = tf.Session()
result = sess.run(hello)
print(result)
#release the resources of sess using 'close()' method
sess.close()

# use the context manager to close sess automatically
with tf.Session() as sess:
    result = sess.run(hello)
    print(result)  
