# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:53:31 2018

@author: lankuohsing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:56:46 2018
只预测，不训练
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
learn = tf.contrib.learn
HIDDEN_SIZE = 20  # Lstm中隐藏节点的个数
NUM_LAYERS = 1  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度
TRAINING_STEPS = 1000  # 训练轮数
BATCH_SIZE = 32  # batch大小
# In[]
def read_data(data_path='sin.csv',model_path="Models/model_sin3"):

    # In[]
    #读取数据
    f=open(data_path)
    df=pd.read_csv(f)
    data=np.array(df['max'])
    # In[]
    #数据归一化
    normalize_data=(data-np.mean(data))/np.std(data)
    return normalize_data




# In[]


# In[]
# 根据输入序列，切割出输入数据和标签。利用前面的TIMESTEPS项预测后面的一项
def generate_data(seq):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;第i+TIMESTEPS项作为输出
    # 即用sin函数前面的TIMESTPES个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])
        Y.append([seq[i + TIMESTEPS]])
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
    output = tf.reshape(outputs[:,TIMESTEPS-1,:], [-1, HIDDEN_SIZE])
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
"""
训练函数
"""
def train_func(train_X,train_y,model_path="Models/model_sin3"):
    if not os.path.exists(model_path):  ###判断文件是否存在，返回布尔值
        os.makedirs(model_path)
    shutil.rmtree(model_path)
    # 进行训练
    # 封装之前定义的lstm
    regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=model_path))
    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    return None


# In[]
"""
测试函数
"""
def test_func(test_X,test_y,model_path="Models/model_sin3"):
    regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=model_path))
    a=regressor.score(x=test_X, y=test_y)
    predicted_data = [[pred] for pred in regressor.predict(test_X)]
    return predicted_data






# In[]
"""
下面的代码是我用来调试用的，不用管，实际调用时不会调用它们
"""
if __name__=="__main__":
    # 生成数据
    data_path='sin.csv'
    model_path="Models/model_sin3"
    normalize_data=read_data(data_path=data_path,model_path=model_path)
    train_X, train_y = generate_data(normalize_data[0:5000])
    test_X, test_y = generate_data(normalize_data[5000:10000])
    train_X=np.transpose(train_X,[0,2,1])
    test_X=np.transpose(test_X,[0,2,1])
    train_func(train_X,train_y,model_path=model_path)
    # In[]
    predicted_data=test_func(test_X,test_y,model_path=model_path)
    # In[]
    print(predicted_data)
    rmse = np.sqrt(((predicted_data - test_y) ** 2).mean(axis=0))
    print("Mean Square Error is:%f" % rmse[0])
    # In[]
    plot_test, = plt.plot(test_y, label='real_sin')
    plot_predicted, = plt.plot(predicted_data, label='predicted_data')
    plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
    x_start=5000
    x_end=5100
    y_start=-2
    y_end=4
    #plt.axis([x_start,x_end,y_start,y_end])
    plt.show()