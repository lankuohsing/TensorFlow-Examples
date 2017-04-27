# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:26:18 2017

@author: lankuohsing
"""
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython
# matplotlib inline

from sklearn import datasets
from sklearn.cross_validation import train_test_split
digits = datasets.load_digits()
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)

image_size = 8
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(Xtrain, ytrain)
test_dataset, test_labels = reformat(Xtest, ytest)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
batch_size = train_dataset.shape[0]
hidden_units = 1024
    
graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

    # Stage 1 - Training computation.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
    biases1 = tf.Variable(tf.zeros([hidden_units]))
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)

    # Final stage
    weights2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(hidden1, weights2) + biases2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), 
              weights2) + biases2)
    num_steps = 3001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : train_dataset, tf_train_labels : train_labels}
        _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, train_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
  
  
  
  