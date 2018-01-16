# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:56:46 2018
根据一维时间序列信息，预测股票；利用前TIMESTEPS个点预测下一个点
@author: lankuohsing
"""

# In[]
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
# In[]
import shutil
import os

model_path="Models/model_3_2"
if not os.path.exists(model_path):  ###判断文件是否存在，返回布尔值
   os.makedirs(model_path)
shutil.rmtree(model_path)

# In[]
#读取数据
f=open('dataset_1.csv')
df=pd.read_csv(f)
data=np.array(df['max'])
#data=data[::-1]
# In[]
#数据归一化
normalize_data=(data-np.mean(data))/np.std(data)

# In[]
learn = tf.contrib.learn
HIDDEN_SIZE = 20  # Lstm中隐藏节点的个数
NUM_LAYERS = 1  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度
TRAINING_STEPS = 10000  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔
# In[]
# 根据输入序列，切割出输入数据和标签。利用前面的TIMESTEPS项预测后面的一项
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE,forget_bias=1.0,state_is_tuple=True)
    return lstm_cell

# 定义lstm模型
def lstm_model(X0, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    print("X0.shape:",X0.shape)
    X=tf.unstack(X0,axis=1)
    print("len(X):",len(X))
    print("X[0].shape",X[0].shape)
    outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)
    print("outputs.shape:",outputs[-1].shape)
    output = tf.reshape(outputs[-1], [-1, HIDDEN_SIZE])
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    #注意，这里不用在最后加一层softmax层，因为不是分类问题
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=0.1)
    return predictions, loss, train_op
# In[]
X,Y=generate_data(normalize_data)
#X=np.transpose(X,[0,2,1])
# In[]
# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=model_path))
# 生成数据
train_X, train_y = generate_data(normalize_data[0:5000])
#train_X=np.transpose(train_X,[0,2,1])
#train_y=np.transpose(train_X,[0,2,1])
test_X, test_y = generate_data(normalize_data[5000:6000])
#test_X=np.transpose(test_X,[0,2,1])
#test_y=np.transpose(test_X,[0,2,1])
# 拟合数据
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 计算预测值
# In[]
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])

# In[]
plot_test, = plt.plot(test_y, label='real_sin')
plot_predicted, = plt.plot(predicted, label='predicted')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
x_start=600
x_end=800
y_start=-1
y_end=-0.2
plt.axis([x_start,x_end,y_start,y_end])
plt.show()