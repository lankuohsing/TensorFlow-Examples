# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:58:10 2021

@author: lankuohsing
"""
# https://www.jb51.net/article/180139.htm
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
# In[]
model_dir="models"
pbName = 'graphA1.pb'
tf.reset_default_graph()
def graphCreate() :
    with tf.Session() as sess :
        var1 = tf.placeholder ( tf.int32 , name='var1' )
        var2 = tf.Variable( 20 , name='var2' )#实参name='var2'指定了操作名，该操作返回的张量名是在
                           #'var2'后面:0 ,即var2:0 是返回的张量名，也就是说变量
                           # var2的名称是'var2:0'

        var3 = tf.Variable( 30 , name='var3' )
        var4 = tf.Variable( 40 , name='var4' )
        var4op = tf.assign( var4 , 1000 , name = 'var4op1' )
        sum = tf.Variable( 4, name='sum' )
        sum = tf.add ( var1 , var2, name = 'var1_var2' )
        sum = tf.add( sum , var3 , name='sum_var3' )
        sumOps = tf.add( sum , var4 , name='sum_operation' )
        oper = tf.get_default_graph().get_operations()
        with open(os.path.join(model_dir, 'operation1.csv'),'wt' ) as f:
            s = 'name,type,output\n'
            f.write( s )
            for o in oper:
                s = o.name
                s += ','+ o.type
                inp = o.inputs
                oup = o.outputs
                for iip in inp :
                    s #s += ','+ str(iip)
                for iop in oup :
                    s += ',' + str(iop)

                s += '\n'
                f.write( s )

            for var in tf.global_variables():
                print('variable=> ' , var.name) #张量是tf.Variable/tf.Add之类操作的结果，
                            #张量的名字使用操作名加:0来表示

        init = tf.global_variables_initializer()
        sess.run( init )
        sess.run( var4op )
        print('sum_operation result is Tensor ' , sess.run( sumOps , feed_dict={var1:1}) )

        constant_graph = graph_util.convert_variables_to_constants( sess, sess.graph_def , ['sum_operation'] )
        with open(os.path.join(model_dir,pbName), mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def graphCreateFromPb():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            var1 = tf.placeholder ( tf.int32 , name='var1' )
            var2 = tf.Variable( 2 , name='var2' )#实参name='var2'指定了操作名，该操作返回的张量名是在
                               #'var2'后面:0 ,即var2:0 是返回的张量名，也就是说变量
                               # var2的名称是'var2:0'

            var3 = tf.Variable( 3 , name='var3' )
            var4 = tf.Variable( 4 , name='var4' )
            var4op = tf.assign( var4 , 100 , name = 'var4op1' )
            graph0 = tf.GraphDef()
            with open( os.path.join(model_dir,pbName), mode='rb') as f:
                graph0.ParseFromString( f.read() )
                tf.import_graph_def( graph0 , name = 'pb1' )
            for op in tf.get_default_graph().get_operations():
                print(op.name,"-----",op.values())
            pb_copy_ops=[]
            """
            实际情况中可以通过
            grads_and_vars=optimizer.compute_gradients();
            train_tensor_name=[tensor[1].name for tensor in grads_and_vars]
            来获得需要更新的tensor的名字（一般都是训练过程中需要更新的权重tensor）
            """
            update_op_2=tf.assign(sess.graph.get_tensor_by_name("var2:0"),sess.graph.get_tensor_by_name("pb1/"+"var2:0"))
            update_op_3=tf.assign(sess.graph.get_tensor_by_name("var3:0"),sess.graph.get_tensor_by_name("pb1/"+"var3:0"))
            update_op_4=tf.assign(sess.graph.get_tensor_by_name("var4:0"),sess.graph.get_tensor_by_name("pb1/"+"var4:0"))
            pb_copy_ops=[update_op_2,update_op_3,update_op_4]
            sum = tf.Variable( 4, name='sum' )
            sum = tf.add ( var1 , var2, name = 'var1_var2' )
            sum = tf.add( sum , var3 , name='sum_var3' )
            sumOps = tf.add( sum , var4 , name='sum_operation' )
            init=tf.global_variables_initializer()
            sess.run(init)
            sess.run(var4op)
            sess.run(pb_copy_ops)
            print('sum_operation result is Tensor ' , sess.run( sumOps , feed_dict={var1:1}) )
            constant_graph = graph_util.convert_variables_to_constants( sess, sess.graph_def , ['sum_operation'] )
            with open(os.path.join(model_dir,"graphB1.pb"), mode='wb') as f:
                f.write(constant_graph.SerializeToString())



def graphGet() :
    print("start get:" )
    with tf.Graph().as_default():
        graph0 = tf.GraphDef()
        with open( os.path.join(model_dir,pbName), mode='rb') as f:
            graph0.ParseFromString( f.read() )
            tf.import_graph_def( graph0 , name = '' )

        with tf.Session() as sess :
            init = tf.global_variables_initializer()
            sess.run(init)
            v1 = sess.graph.get_tensor_by_name('var1:0' )
            v2 = sess.graph.get_tensor_by_name('var2:0' )
            v3 = sess.graph.get_tensor_by_name('var3:0' )
            v4 = sess.graph.get_tensor_by_name('var4:0' )
            sumTensor = sess.graph.get_tensor_by_name("sum_operation:0")
            print('sumTensor is : ' , sumTensor )
            print( sess.run( sumTensor , feed_dict={v1:1} ) )
if __name__=="__main__":
    graphCreate()
#    graphGet()
#    graphCreateFromPb()

