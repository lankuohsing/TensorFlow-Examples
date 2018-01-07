# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:29:35 2017
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-12-scope/
@author: lankuohsing
"""
'''
在 Tensorflow 当中有两种途径生成变量 variable, 一种是 tf.get_variable(),
另一种是 tf.Variable(). 如果在 tf.name_scope() 的框架下使用这两种方式, 结果会如下.
'''
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]

'''
可以看出使用 tf.Variable() 定义的时候, 虽然 name 都一样, 但是为了不重复变量名,
Tensorflow 输出的变量名并不是一样的. 所以, 本质上 var2, var21, var22 并不是一样的变量.
而另一方面, 使用tf.get_variable()定义的变量不会被tf.name_scope()当中的名字所影响.
'''