# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:37:39 2017

@author: lankuohsing
"""
# In[]
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
import os
# In[]
data_dir='/tmp/cifar10_data/cifar-10-batches-bin'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
# In[]
for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
# In[]
help(tf.train.shuffle_batch)