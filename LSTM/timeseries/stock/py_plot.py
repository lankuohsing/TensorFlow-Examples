# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:59:06 2018

@author: lankuohsing
"""
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
a=1
plt.plot([1,2,3,4,5])
plt.savefig('table'+str(a)+'.png')
