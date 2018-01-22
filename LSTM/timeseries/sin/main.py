# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:02:32 2018

@author: lankuohsing
"""
# In[]
from sin_predict3 import *

# In[]
normalize_data=read_data(data_path=data_path,model_path=model_path)
# 生成训练数据和测试数据
data_path='sin.csv'
model_path="Models/model_sin3"
normalize_data=read_data(data_path=data_path,model_path=model_path)
train_X, train_y = generate_data(normalize_data[0:5000])
test_X, test_y = generate_data(normalize_data[5000:10000])
train_X=np.transpose(train_X,[0,2,1])
test_X=np.transpose(test_X,[0,2,1])
# In[]训练模型
#train_func(train_X,train_y,model_path=model_path)
# In[]测试模型
predicted_data=test_func(test_X,test_y,model_path=model_path)
# In[]
print(predicted_data)
rmse = np.sqrt(((predicted_data - test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])
