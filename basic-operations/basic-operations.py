# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:47:15 2017
Basic Operations example using TensorFlow library.
@author: lankuohsing
"""
import tensorflow as tf
# basic constant operation
# the value returned by the constructor represnets the outputs
# of the constant op
a = tf.constant(2)
b = tf.constant(3)

# launch the default graph
with tf.Session() as sess:
    print("a=%d, b=%d" % (sess.run(a),sess.run(b)))# 格式化输出
    print("a plus b: %d" % sess.run(a+b))
    print("a multiplied by b: %d" % sess.run(a*b))
    
    
# basic operations with variable as graph input
# the value returned by the constructor represents the output
# of the variable op. (defined as input when running the seesion)
# tf graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#define sonme operations
add = tf.add(a, b)
mul=tf.multiply(a, b)

# launch the default graph
with tf.Session() as sess:
    # run every operation with variable inputs
    print("a=%d, b=%d" % (sess.run(a, feed_dict = {a: 2}), sess.run(b, feed_dict = {b: 3})))
    print("addition with variables: %d" % sess.run(add, feed_dict = {a: 2, b: 3}))
    print("multiplication with variables: %d" % sess.run(mul, feed_dict = {a: 2, b: 3}))
    
matrix1 = tf.placeholder(tf.float32, shape=(1, 2))
matrix2 = tf.placeholder(tf.float32, shape=(2, 1))
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product, feed_dict = {matrix1: [[3., 3.]], matrix2: [[2.], [2.]]})
    print(result)