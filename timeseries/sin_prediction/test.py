# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:52:41 2018

@author: lankuohsing
"""

# In[]
import numpy as np
# In[]
a=np.array([[[1],[2]],
            [[3],[4]],
            [[5],[6]]])

# In[]
b=a.reshape(a.shape[0]*a.shape[1],1)