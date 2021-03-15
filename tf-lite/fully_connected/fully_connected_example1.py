# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:25:14 2021

@author: lankuohsing
"""

# In[]
import numpy as np
import tensorflow as tf

# In[]
def get_model():
    # create a linear regression model
    inputs=tf.keras.Input(shape=(input_size,),name="InputX")
    outputs=tf.keras.layers.Dense(output_size,name="Output")(inputs)
    model=tf.keras.Model(inputs,outputs)
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
np.random.seed(1)
input_size=1024
output_size=8192
input_list=np.random.randint(0,10,input_size).tolist()
output_list=np.random.randint(0,10,output_size).tolist()
# In[]
model=get_model()
train_X=tf.constant([input_list])
train_Y=tf.constant([output_list])
dataset=tf.data.Dataset.from_tensors((train_X,train_Y))


# In[]
history=model.fit(dataset,epochs=100,validation_data=dataset,validation_steps=10)

pred=model.predict(train_X)

# In[]
keras2pb(model,"./models/fc1/","fc_model.pb")
model.save("./models/fc1/")
# In[]
converter=tf.lite.TFLiteConverter.from_saved_model("./models/fc1")
tflite_model=converter.convert()
open("./models/fc1_model.tflite","wb").write(tflite_model)