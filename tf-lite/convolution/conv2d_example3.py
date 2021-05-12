# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:07:05 2021

@author: lankuohsing
"""

"""
用代码例子证明1×1卷积和FC在输入的H×W为1×1时是等价的
"""
import numpy as np
import tensorflow as tf
# In[]
def get_model():
    # create a linear regression model
    model=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape[1:],name="InputX"),
                               tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=(1, 1),
                                                      padding='valid', activation=None, use_bias=False,
                                                      input_shape=input_shape[1:])])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
def keras2pb(keras_model,model_path,pb_name):
    from tensorflow.python.keras.saving import saving_utils as _saving_utils
    from tensorflow.python.framework import convert_to_constants as _convert_to_constants
    from tensorflow.compat.v1 import graph_util
    func=_saving_utils.trace_model_call(keras_model)
    concrete_func=func.get_concrete_function()
    frozen_func, graph_def=(_convert_to_constants.convert_variables_to_constants_v2_as_graph(concrete_func,lower_control_flow=False))
    graph=graph_util.remove_training_nodes(graph_def)
    tf.io.write_graph(graph,model_path,pb_name,as_text=False)
    return
# In[]
# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
np.random.seed(1)
input_shape = (1, 1, 1, 3)#[B,H,W,C]
output_shape=(1,1,1,3)#[B,H,W,C]
#x = tf.random.normal(input_shape)
#y = tf.keras.layers.Conv2D(
#filters=2, kernel_size=2, strides=(1, 1), padding='valid', activation=None, use_bias=False, input_shape=input_shape[1:])(x)
# In[]
input_list=np.random.randint(0,10,input_shape).tolist()
output_list=np.random.randint(0,10,output_shape).tolist()
# In[]
model=get_model()
train_X=tf.constant(input_list)
train_Y=tf.constant([output_list])
dataset=tf.data.Dataset.from_tensors((train_X,train_Y))
# In[]
history=model.fit(dataset,epochs=100,validation_data=dataset,validation_steps=10)

pred=model.predict(train_X)

# In[]
keras2pb(model,"./models/conv3/","fc_model.pb")
model.save("./models/conv3/")
# In[]
model=tf.saved_model.load("./models/conv3")
concrete_func=model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape(input_shape)
converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model=converter.convert()
with open("./models/conv3_model.tflite","wb")as wf:
    wf.write(tflite_model)
# In[]
weights =np.load("./models/conv3_weights.npy")
# In[]
input_np=np.array(input_list[0][0][0])
pred_np=pred.reshape(pred.shape[3],)
weights_np=weights.reshape(weights.shape[0],weights.shape[3])
# In[]
fc_result=np.dot(weights_np,input_np)