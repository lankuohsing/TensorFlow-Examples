# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:17:06 2017

@author: lankuohsing
"""
import tensorflow as tf  

#case 1  
'''
1.考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），
用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，
最后会得到一张3×3的feature map
'''
input = tf.Variable(tf.random_normal([1,3,3,1]))  
filter = tf.Variable(tf.random_normal([1,1,1,1]))  

op1 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID') 

#case 2  
input = tf.Variable(tf.random_normal([1,3,3,5]))  
filter = tf.Variable(tf.random_normal([1,1,5,1]))  
  
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
#case 3  
input = tf.Variable(tf.random_normal([1,3,3,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
#case 4  
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
#case 5  
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  
#case 6  
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
  
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  
#case 7  
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
  
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')  
#case 8  
input = tf.Variable(tf.random_normal([10,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
  
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')  

init = tf.initialize_all_variables()  
with tf.Session() as sess:  
    sess.run(init)  
    print("case 1")  
    print(sess.run(op1))   
    
    print("case 2")  
    print(sess.run(op2))  
    print("case 3")  
    print(sess.run(op3))  
    print("case 4")  
    print(sess.run(op4))  
    print("case 5")  
    print(sess.run(op5))  
    print("case 6")  
    print(sess.run(op6))  
    print("case 7")  
    print(sess.run(op7))  
    print("case 8")  
    print(sess.run(op8)) 
    
