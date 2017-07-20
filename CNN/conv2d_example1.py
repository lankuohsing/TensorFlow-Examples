# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:35:03 2017

@author: lankuohsing
"""

import tensorflow as tf  

#case 1  
'''
1.考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），
用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，
最后会得到一张3×3的feature map
'''
input1 = tf.constant([[[[1.],[2.],[3.]],[[4.],[5.],[6.]],[[7.],[8.],[9.]]]])  
filter1 = tf.constant([[[[10., 100]]]]) 
op1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')

#case 2  
'''
2.增加图片的通道数，使用一张3×3三通道的图像（对应的shape：[1，3，3，3]），
用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，
仍然是一张3×3的feature map，这就相当于每一个像素点，卷积核都与该像素点的每一个通道做点积
'''
input2 = tf.constant([[[[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]],
                       [[4.,4.,4.],[5.,5.,5.],[6.,6.,6.]],
                       [[7.,7.,7.],[8.,8.,8.],[9.,9.,9.]]]]) 
filter2 = tf.constant([[[[10.],[20.],[30.]]]]) 
op2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')
#case 3  
'''
1.现在有一张5×5单通道的图像（对应的shape：[1，5，5，1]），
用一个3×3的卷积核（对应的shape：[3，3，1，1]）去做卷积，
最后会得到一张3×3的feature map
'''
input3 = tf.constant([[[[1.],[2.],[3.],[2.],[3.]],
                       [[4.],[5.],[6.],[5.],[6.]],
                       [[7.],[8.],[9.],[8.],[9.]],
                       [[7.],[8.],[9.],[8.],[9.]],
                       [[7.],[8.],[9.],[8.],[9.]]]])  
filter3 = tf.constant([[[[10.]],[[20.]],[[30.]]],
                       [[[40.]],[[50.]],[[60.]]],
                       [[[70.]],[[80.]],[[90.]]]]) 
op3 = tf.nn.conv2d(input3, filter3, strides=[1, 1, 1, 1], padding='SAME')
#case 4  
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
  
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID') 
init = tf.initialize_all_variables()  
with tf.Session() as sess:  
    sess.run(init)  
    print("case 1")  
    print("input1: ",sess.run(input1),"shape: ",sess.run(tf.shape(input1)))  
    print("filter1: ",sess.run(filter1),"shape: ",sess.run(tf.shape(filter1)))  
    print("results1: ",sess.run(op1),"shape: ",sess.run(tf.shape(op1))) 
    
    print("case 2")  
    print("input2: ",sess.run(input2),"shape: ",sess.run(tf.shape(input2)))  
    print("filter2: ",sess.run(filter2),"shape: ",sess.run(tf.shape(filter2)))  
    print("results2: ",sess.run(op2),"shape: ",sess.run(tf.shape(op2))) 
    
    print("case 3")  
    print("input3: ",sess.run(input3),"shape: ",sess.run(tf.shape(input3)))  
    print("filter3: ",sess.run(filter3),"shape: ",sess.run(tf.shape(filter3)))  
    print("results3: ",sess.run(op3),"shape: ",sess.run(tf.shape(op3))) 
    