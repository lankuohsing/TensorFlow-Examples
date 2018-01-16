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
from tensorflow.contrib import rnn

learn=tf.contrib.learn
hidden_size=30
num_layers=1 #number of LSTM layers
Times_Steps=10
Training_Steps=5000
Trainging_examples=1000
Test_examples=369
BATCH_SIZE=100
# In[]

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-Times_Steps-5):
        X.append([seq[i:i+Times_Steps]])
        y.append([seq[i+Times_Steps+5]])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

def lstm_model(X,y):
    #使用多层的lstm结构
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)
    print("X.shape:",X.shape)
    x_=tf.unstack(X,axis=1)
    print("x_.shape:",x_[0].shape)
    output, _=rnn.static_rnn(cell, x_, dtype=tf.float32)
    output=output[-1]
    prediction,loss=learn.models.linear_regression(output,y)
    train_op=tf.contrib.layers.optimize_loss(
            loss,tf.contrib.framework.get_global_step(),
            optimizer="Adagrad",learning_rate=0.1)
    return prediction, loss, train_op
# In[]
#导入数据
Data_x_File="UDDS_speed.xlsx"
book= xlrd.open_workbook (Data_x_File, encoding_override = "utf-8")
data_sheet = book.sheet_by_index(0)
data= np.asarray([data_sheet.col_values(0)])
data=data.T
M=len(data)
train_X,train_y=generate_data(data[0:Trainging_examples,0])
test_X,test_y=generate_data(data[Trainging_examples:M,0])
# In[]
#建立深层循环网络模型
regressor= learn.Estimator(model_fn=lstm_model)

#调用fit函数训练模型
regressor.fit(train_X,train_y,batch_size=BATCH_SIZE,steps=Training_Steps)

predicted_train=[[pred] for pred in regressor.predict(train_X)]
rmse_train=np.sqrt(((predicted_train-train_y)**2).mean(axis=0))

predicted=[[pred] for pred in regressor.predict(test_X)]
rmse=np.sqrt(((predicted-test_y)**2).mean(axis=0))
print("Mean Square Error is:%f"%rmse[0])

# In[]

#绘图输出
T1=range(1,990-4)
T2=range(991-4,1349-8)
fig=plt.figure()
plt.plot(T1,predicted_train,'r',T2,predicted,'r')
plt.plot(T1,train_y,'g',T2,test_y,'b')
#plt.legend([plot_predicted,plot_test],['predicted','real_speed'])
plt.xlabel("Time [s]",fontsize=35)
plt.ylabel("Velocity [Km/h]",fontsize=35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.set_size_inches(20, 7.5)
plt.show()
fig.savefig('UDDS_hidden_50.png')



