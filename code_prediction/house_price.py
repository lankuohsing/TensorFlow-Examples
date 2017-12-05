# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:10:22 2017

@author: lankuohsing
"""

# In[]
import tensorflow as tf
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
# In[]
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

# In[]
def autoNorm(data):         #传入一个矩阵
    mins = data.min(0)      #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)      #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins    #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))     #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]                     #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData
# In[]
# In[]
# 1.训练的数据
# Make up some real data
#x_data = np.linspace(-1,1,300)[:, np.newaxis]
# 1.训练的数据
# Make up some real data
Data_x_File="dataset_for_simulation.xlsx"
#Data_y_File= "F:\learning\G_NO1\Courses\machine_learning\programming\math_function\data\data_y.xls"
book= xlrd.open_workbook (Data_x_File, encoding_override = "utf-8")
training_sheet = book.sheet_by_index(0)
test_sheet = book.sheet_by_index(1)
#x_data = np.asarray([sheet.row_values(i) for i in range(0,sheet.nrows)])  #输出完整矩阵
#x_input1= np.asarray([sheet.col_values(0)])#输出第二列
x_data= np.asarray([training_sheet .col_values(i) for i in range(7,11)])
x_data=x_data.T
y_data =np.asarray([training_sheet .col_values(12)])
y_data=y_data.T

x_test= np.asarray([test_sheet.col_values(i) for i in range(7,11)])
x_test=x_test.T
y_test=np.asarray([test_sheet.col_values(12)])
y_test=y_test.T
# In[]
x_data_norm=autoNorm(x_data)
x_test_norm=autoNorm(x_test)


# In[]
x_data.shape[0]
# In[]
batch_indices=nr.choice(range(num_train),BATCH_SIZE)
batch_x=x_data_norm[batch_indices,:]
batch_y=y_data[batch_indices,:]
xs=batch_x
ys=batch_y
# In[]
print(x_data_norm.shape)
print(y_data.shape)
print(xs.shape)
print(ys.shape)
# In[]


