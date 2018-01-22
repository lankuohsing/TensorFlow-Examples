# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:21:47 2018

@author: lankuohsing
"""
# In[]
import shutil
import os
# In[]
model_path="Models/model_sin3/"
a=[]
# In[]
for root, dirs, files in os.walk(model_path, topdown=True, onerror=None, followlinks=False):
    a.append(files)
    print(files)
# In[]
for i in range(0,len(a[0])):
    os.remove(model_path++a[0][i])
# In[]
for i in range(0,len(a[1])):
    os.remove(model_path+'eval_score/'+a[1][i])