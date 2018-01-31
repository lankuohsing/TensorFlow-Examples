# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:29:52 2018

@author: lankuohsing
"""


MODEL_PATH="Models/model_sin3"


# In[]
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#from tensorflow.contrib import predictor
mpl.use('Agg')
from matplotlib import pyplot as plt

"""
if not os.path.exists(MODEL_PATH):  ###判断文件是否存在，返回布尔值
   os.makedirs(MODEL_PATH)
shutil.rmtree(MODEL_PATH)
"""
# In[]
#读取数据
f=open('sin.csv')
df=pd.read_csv(f)
data=np.array(df['value'])
#data=data[::-1]
# In[]
#数据归一化
#normalize_data=(data-np.mean(data))/np.std(data)
normalize_data=data
# In[]
learn = tf.contrib.learn
HIDDEN_SIZE = 20  # Lstm中隐藏节点的个数
NUM_LAYERS = 1  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度
TRAINING_STEPS = 1000  # 训练轮数
BATCH_SIZE = 32  # batch大小

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
def lstm_model(X):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    print("X.shape:",X.shape)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    print("outputs.shape:",outputs.shape)
    #print("final_state.shape:",final_state[0].dtype)
    output = tf.reshape(outputs[:,TIMESTEPS-1,:], [-1, HIDDEN_SIZE])
    return output
def lstm_train_op(X,y=None):
    output=lstm_model(X)
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    #注意，这里不用在最后加一层softmax层，因为不是分类问题
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
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
regressor = SKCompat(learn.Estimator(model_fn=lstm_train_op, model_dir=MODEL_PATH))
#regressor = learn.Estimator(model_fn=lstm_model, model_dir=MODEL_PATH)
#predict_fn = predictor.from_saved_model(MODEL_PATH)
# 生成数据
train_X, train_y = generate_data(normalize_data[0:5000])
test_X, test_y = generate_data(normalize_data[5000:10000])
train_X=np.transpose(train_X,[0,2,1])
test_X=np.transpose(test_X,[0,2,1])
# 拟合数据
#regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 计算预测值
# In[]
#predicted = [[pred] for pred in regressor.predict(test_X)]
regressor.score(test_X,test_y)
predicted = list(regressor.predict(test_X))

# 计算MSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is:%f" % rmse[0])

# In[]
figure1=plt.figure(1)
figure1.set_figheight(5)
figure1.set_figwidth(8)
plot_test, = plt.plot(test_y, label='real_sin')
plot_predicted, = plt.plot(predicted, label='predicted')
plt.legend([plot_predicted, plot_test],['predicted', 'real_sin'])
x_start=5000
x_end=5100
y_start=-2
y_end=4
#plt.axis([x_start,x_end,y_start,y_end])
plt.show()
# In[]
a=np.zeros((3,1))
a
