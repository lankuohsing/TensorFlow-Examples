# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:15:48 2020

@author: lankuohsing
"""

# In[]
import numpy as np
import tensorflow as tf#本代码采用的tf2.3版本
import matplotlib.pyplot as plt
# In[]
# 1. 定义RNN的参数。
HIDDEN_SIZE = 10                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 10000                   # 训练数据个数。
TESTING_EXAMPLES = 1000                     # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。
# In[]
# 2. 产生正弦数据。
def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
def plot_figure(predictions,labels,figure_name):
    #对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predict_sin')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.title(figure_name)
    plt.savefig("figures/"+figure_name+"_tf2.png")
    plt.show()
# In[]
# 用正弦函数生成训练和测试数据集合。
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
sin_input_train=np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)
sin_input_test=np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)

# In[]
train_X, train_y = generate_data(np.sin(sin_input_train))
test_X, test_y = generate_data(np.sin(sin_input_test))

# In[]
# 5. 执行训练和测试。
# 将训练数据以数据集的方式提供给计算图。
train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))#返回的是DataSet对象
#repeat(count)表示构建count个epoch
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)#取出1000个样本并打乱；每次取出数量为BATCH_SIZE的样本作为一个batch;不重复；
test_ds = tf.data.Dataset.from_tensor_slices((test_X,test_y))#返回的是DataSet对象
#repeat(count)表示构建count个epoch
test_ds = test_ds.shuffle(1000).batch(BATCH_SIZE)#取出1000个样本并打乱；每次取出数量为BATCH_SIZE的样本作为一个batch;不重复；
# In[]
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10),
    tf.keras.layers.Dense(1)
])
# In[]
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(1e-4))
predictions = model.predict(test_X)
plot_figure(predictions,test_y,"before_training")
# In[]
history = model.fit(train_ds, epochs=10,
                    validation_data=train_ds,
                    validation_steps=1000)
# In[]
test_loss = model.evaluate(test_ds)
print('Test Loss: {}'.format(test_ds))
# In[]
predictions = model.predict(test_ds)
plot_figure(predictions,test_y,"afer_training")