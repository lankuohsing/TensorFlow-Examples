# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 23:43:18 2021

@author: lankuohsing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# In[]
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
writer = tf.summary.FileWriter('./log')
writer.add_graph(tf.get_default_graph())
writer.flush()
# open anaconda prompt and enter the log dir. then tensorboard --logdir ./
