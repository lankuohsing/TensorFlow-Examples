# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:10:25 2018

@author: janti
"""
# In[]
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import xlrd
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')

# In[]
import shutil
import os

model_path="model"
if not os.path.exists(model_path):  ###判断文件是否存在，返回布尔值
   os.makedirs(model_path)
   
shutil.rmtree(model_path)

# In[]
#from tensorflow.contrib import rnn 

def generate_data(seq): 
    X=[]
    Y=[]
    m=int(len(seq)/Times_Steps)-1
    for i in range(m):
        
        X.append([seq[i*Times_Steps:i*Times_Steps+Times_Steps]])
        Y.append([seq[i*Times_Steps+Times_Steps:i*Times_Steps+2*Times_Steps]]) 
    return np.array(X,dtype=np.float32), np.array(Y,dtype=np.float32)
            

def lstm_model(X,Y):
    #使用多层的lstm结构
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)
    #x_=tf.unstack(X,axis=1)
    #output, _=rnn.static_rnn(cell, x_, dtype=tf.float32) 
   # output=output[-1]
    outputs, _=tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(outputs[:,:,:], [-1, hidden_size])
    prediction,loss=learn.models.linear_regression(output,Y)
    train_op=tf.contrib.layers.optimize_loss(
            loss,tf.contrib.framework.get_global_step(),
            optimizer="Adagrad",learning_rate=0.1)
    return prediction, loss, train_op

# In[]
#导入数据
    
learn=tf.contrib.learn
hidden_size=100
num_layers=1 #number of LSTM layers
Times_Steps=10
Training_Steps=50000
Trainging_examples=5010
BATCH_SIZE=100
    
# In[]
# 进行训练
# 封装之前定义的lstm
#建立深层循环网络模型
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=model_path))
Data_x_File="JC08_velocity_RNN.xlsx"
book= xlrd.open_workbook (Data_x_File, encoding_override = "utf-8")
data_sheet = book.sheet_by_index(0)
data= np.asarray([data_sheet.col_values(0)])
data=data.T
M=len(data)
train_X,train_Y=generate_data(data[0:Trainging_examples,0])
test_X,test_Y=generate_data(data[Trainging_examples:M,0])


#regressor= learn.Estimator(model_fn=lstm_model)
#调用fit函数训练模型
regressor.fit(train_X,train_Y,batch_size=BATCH_SIZE,steps=Training_Steps)

predicted_train=[[pred] for pred in regressor.predict(train_X)]
predicted_train=np.array(predicted_train).reshape((-1,1))
train_Y=np.array(train_Y).reshape((-1,1))
rmse_train=np.sqrt(((predicted_train-train_Y)**2).mean(axis=0))
#saver = tf.train.Saver(write_version=tf.train.SaverDef.V2) # 声明tf.train.Saver类用于保存模型
#saver_path = saver.save(regressor.predict, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件

predicted=[[pred] for pred in regressor.predict(test_X)]
predicted=np.array(predicted).reshape((-1,1))
test_Y=np.array(test_Y).reshape((-1,1))
rmse=np.sqrt(((predicted-test_Y)**2).mean(axis=0))   #.sum(axis=0)
print("Mean Square Error is:%f"%rmse[0])


#绘图输出
T1=range(1,5001)
T2=range(5001,6001)

fig=plt.figure()
plt.plot(T1,predicted_train,'r',T2,predicted,'r')
plt.plot(T1,train_Y,'g',T2,test_Y,'b')#,
#plt.legend([plot_predicted,plot_test],['predicted','real_speed'])
plt.xlabel("Time [s]",fontsize=35)
plt.ylabel("Velocity [Km/h]",fontsize=35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.set_size_inches(20, 7.5)
plt.show()
fig.savefig('JC08_H10.png')



