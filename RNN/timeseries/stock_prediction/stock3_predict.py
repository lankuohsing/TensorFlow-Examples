# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:02:04 2018

@author: lankuohsing
"""


# In[]
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
# In[]
import shutil
import os
#模型存储路径
MODEL_PATH="Models/model_stock3"
"""
if not os.path.exists(MODEL_PATH):  ###判断文件是否存在，返回布尔值
   os.makedirs(MODEL_PATH)
shutil.rmtree(MODEL_PATH)
"""
# In[]
#读取数据
f=open('stock_2.csv')
df=pd.read_csv(f)
data=df.iloc[:,2:10].values
#data=data[::-1]
# In[]
data_length=data.shape[0]
train_length=int(data_length*0.7)
test_length=data_length-train_length
# In[]
#数据归一化
#normalize_data=(data-np.mean(data))/np.std(data)
feature_range=(0,1)
scaler = MinMaxScaler(copy=True,feature_range=feature_range)#copy=True保留原始数据矩阵
normalize_data=scaler.fit_transform(data)
# In[]
"""
Hyperparameters
"""
learn = tf.contrib.learn
HIDDEN_SIZE = 30  # Lstm中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度，也即input sequence的长度
TRAINING_STEPS = 1000  # 训练轮数
BATCH_SIZE = 200  # batch大小
PREDICT_STEPS=5 #每一轮的预测点个数，也即output sequence长度
NUM_FEATURES=7#输入特征维数
# In[]
# 根据输入序列，切割出输入数据和标签。利用前面的TIMESTEPS项预测后面的PREDICT_STEPS项
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;
    # 第i+TIMESTEPS项和后面的PREDICT_STEPS-1项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测后面的PREDICT_STEPS个点的值
    for i in range(len(seq) - TIMESTEPS -(PREDICT_STEPS-1)):
        X.append(seq[i:i + TIMESTEPS,0:NUM_FEATURES])
        Y.append([seq[i + TIMESTEPS:i + TIMESTEPS+PREDICT_STEPS,3]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE,forget_bias=1.0,state_is_tuple=True)
    return lstm_cell

# 定义lstm模型
def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    print("X.shape:",X.shape)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    print("outputs.shape:",outputs.shape)
    #print("final_state.shape:",final_state[0].dtype)
    output = tf.reshape(outputs[:,TIMESTEPS-PREDICT_STEPS:TIMESTEPS,:], [-1, HIDDEN_SIZE])
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    #注意，这里不用在最后加一层softmax层，因为不是分类问题
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    print("predictions.shape:",predictions.shape)
    print("labels.shape:",labels.shape)
    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=0.1)
    return predictions, loss, train_op
# In[]
# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=MODEL_PATH))
#regressor = learn.Estimator(model_fn=lstm_model, model_dir=MODEL_PATH)
# 生成数据
train_X, train_y = generate_data(normalize_data[0:train_length,:])
test_X, test_y = generate_data(normalize_data[train_length:data_length,:])
# In[]
#train_X=np.transpose(train_X,[0,2,1])
train_y=np.transpose(train_y,[0,2,1])
#test_X=np.transpose(test_X,[0,2,1])
test_y=np.transpose(test_y,[0,2,1])
# 拟合数据
# In[]
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# In[]
def final_data_for_plot(predicted_list,test_y):

    test_y_list=test_y.reshape(test_y.shape[0]*test_y.shape[1],1).tolist()

    final_predicted_list=[]
    final_test_y_list=[]
    for i in range(0,len(predicted_list)-PREDICT_STEPS+1):
        if i%(PREDICT_STEPS*PREDICT_STEPS)==0:
            final_predicted_list.extend(predicted_list[i:i+PREDICT_STEPS])
            final_test_y_list.extend(test_y_list[i:i+PREDICT_STEPS])
    final_predicted=np.array(final_predicted_list).reshape(len(final_predicted_list),1)
    final_test_y=np.array(final_test_y_list).reshape(len(final_test_y_list),1)
    return final_predicted, final_test_y
# 计算预测值
# In[]
#predicted = [[pred] for pred in regressor.predict(test_X)]
regressor.score(test_X,test_y)
predicted_list = list(regressor.predict(test_X))


# In[]
final_predicted, final_test_y=final_data_for_plot(predicted_list,test_y)
# In[]
final_predicted=(final_predicted-feature_range[0])/(feature_range[1]-feature_range[0])\
*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]
final_test_y=(final_test_y-feature_range[0])/(feature_range[1]-feature_range[0])\
*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]
# In[]
# 计算MSE
rmse = np.sqrt(((final_predicted - final_test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])

# In[]
figure1=plt.figure(1)
figure1.set_figheight(5)
figure1.set_figwidth(8)
plot_test, = plt.plot(final_test_y, label='real_sin')
plot_predicted, = plt.plot(final_predicted, label='predicted')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
x_start=1000
x_end=1060
y_start=-1
y_end=-0.2
#plt.axis([x_start,x_end,y_start,y_end])
plt.savefig('figures/test_'+'TIMESTEPS='+str(TIMESTEPS)+'PREDICT_STEPS='+str(PREDICT_STEPS)+'.png')
plt.show()
# In[]
predicted_list = list(regressor.predict(train_X))
final_predicted, final_test_y=final_data_for_plot(predicted_list,train_y)
# In[]
final_predicted=(final_predicted-feature_range[0])/(feature_range[1]-feature_range[0])\
*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]
final_test_y=(final_test_y-feature_range[0])/(feature_range[1]-feature_range[0])\
*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]
# In[]
# 计算MSE
rmse = np.sqrt(((final_predicted - final_test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])
# In[]
figure1=plt.figure(1)
figure1.set_figheight(5)
figure1.set_figwidth(8)
plot_test, = plt.plot(final_test_y, label='real_sin')
plot_predicted, = plt.plot(final_predicted, label='predicted')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
x_start=1000
x_end=2000
y_start=100
y_end=1500
plt.axis([x_start,x_end,y_start,y_end])
plt.savefig('figures/train_'+'TIMESTEPS='+str(TIMESTEPS)+'PREDICT_STEPS='+str(PREDICT_STEPS)+'.png')
plt.show()
