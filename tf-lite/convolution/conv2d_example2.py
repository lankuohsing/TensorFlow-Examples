# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:54:34 2021

@author: lankuohsing
"""

import numpy as np
import tensorflow as tf
# In[]
x_in = np.array([
        [
  [[2], [1], [2], [0], [1]],
  [[1], [3], [2], [2], [3]],
  [[1], [1], [3], [3], [0]],
  [[2], [2], [0], [1], [1]],
  [[0], [0], [3], [1], [2]], ]
    ])#[batch_shape,in_height, in_width, in_channels]=[1,5,5,1]
kernel_in = np.array([
 [ [[2, 0.1]], [[3, 0.2]] ],
 [ [[0, 0.3]],[[1, 0.4]] ],
 ])#[filter_height, filter_width, in_channels, out_channels]=[2,2,1,2]
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
y=tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
# In[]
y_out=y.numpy()
