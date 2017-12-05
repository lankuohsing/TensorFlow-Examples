# -*- coding: utf-8 -*-
"""
Created on Wed Dec 1 09:51:17 2017

@author: janti
"""
# In[]
import tensorflow as tf
import numpy as np
from numpy import random as nr
import matplotlib.pyplot as plt
import xlrd

# In[]
LEARNING_RATE_BASE = 0.08
LEARNING_RATE_DECAY = 0.98
total_steps=10000
DECAY_RATE=100#每隔一定步数，学习率衰减一次
BATCH_SIZE=500
# In[]
# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # 区别：大框架，定义层 layer，里面有 小部件
    with tf.name_scope('layer'):
     # 区别：小部件
     with tf.name_scope('weights'):
         Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
     with tf.name_scope('biases'):
         biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
     with tf.name_scope('Wx_plus_b'):
         Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
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
x_data_norm=autoNorm(x_data)
x_test_norm=autoNorm(x_test)
num_train=x_data_norm.shape[0]
# In[]
# 2.定义节点准备接收数据
# define placeholder for inputs to network
# 区别：大框架，里面有 inputs x，y
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 4],name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')
    y_prediction = tf.placeholder(tf.float32, [None, 1],name='y_prediction')
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(xs,4,50, activation_function=tf.sigmoid)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
l2= add_layer(l1, 50, 15, activation_function=tf.sigmoid)#
#l3= add_layer(l2, 10, 5, activation_function=tf.sigmoid)#
prediction = add_layer(l2, 15, 1, activation_function=None)#
# In[]
# 4.定义 loss 表达式
# the error between prediciton and real data

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# In[]
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, DECAY_RATE, LEARNING_RATE_DECAY)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
writer = tf.summary.FileWriter("D:/neural_1_logs", sess.graph)
#writer = tf.train.SummaryWriter('/root/tensor-board/logs/mnist_logs', sess.graph)
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(total_steps):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    batch_indices=nr.choice(range(num_train),BATCH_SIZE)
    batch_x=x_data_norm[batch_indices,:]
    batch_y=y_data[batch_indices,:]
    xs1=batch_x
    ys1=batch_y
    sess.run(train_step, feed_dict={xs: xs1, ys: ys1})
    if i % 500 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: xs1, ys: ys1}))

# 1.测试的数据
y_prediction = sess.run(prediction, feed_dict={xs: x_data_norm})
plt.figure()
plt.plot(y_data,'b')#,x_data[:,1],y_prediction,'r'
plt.plot(y_prediction,'r')
plt.show()
