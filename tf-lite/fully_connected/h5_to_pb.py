# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:59:58 2021

@author: lankuohsing
"""

# https://www.cnblogs.com/chenzhen0530/p/10685944.html

"""
将keras的.h5的模型文件，转换成TensorFlow的pb文件
在tf2.5下得到验证

"""
# ==========================================================

from keras.models import load_model
import tensorflow as tf
import os
from keras import backend


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """.h5模型文件转换成pb模型文件
    Argument:
        h5_model: str
            .h5模型文件
        output_dir: str
            pb模型文件保存路径
        model_name: str
            pb模型文件名称
        out_prefix: str
            根据训练，需要修改
        log_tensorboard: bool
            是否生成日志文件
    Return:
        pb模型文件
    """
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        print(h5_model.outputs[i].name)
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    tf.import_graph_def(init_graph,name="")
    for op in tf.compat.v1.get_default_graph().get_operations():
        for value in op.values():
            print(value.name)
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    # 输出日志文件
#    if log_tensorboard:
#        from tensorflow.python.tools import import_pb_to_tensorboard
#        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)


if __name__ == '__main__':
    with tf.Graph().as_default():
        #  .h模型文件路径参数
        input_path = './models/fc2/'
        model_file = 'fc2.h5'
        model_file_path = os.path.join(input_path, model_file)
        output_graph_name = model_file[:-3] + '.pb'

        #  pb模型文件输出输出路径
        output_dir = './models/fc2/'

        #  加载模型
        h5_model = load_model(model_file_path)
        h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
        print('Finished')