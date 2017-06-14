from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# a few activation functions to play with
def halfSquare(x):
    return tf.square(tf.nn.relu(x))

def myIdty(x):
    return x

act_list = {'sigmoid': tf.nn.sigmoid,     \
            'tanh': tf.nn.tanh,        \
            'relu': tf.nn.relu,        \
            'abs': tf.abs,         \
            'square': tf.square,      \
            'half_square': halfSquare, \
            'identity': myIdty}

def initWeight(shape, stddev=0.1): 
    W = tf.truncated_normal(shape, stddev=stddev)
    return W

def initBias(shape, bias=0.1):
    b = tf.constant(bias, shape=shape)
    return b

def initWeightBias(shape, stddev=0.1, bias=0.1):
    return initWeight(shape, stddev), initBias(shape[-1:], bias)

class ConvLayer(object):
    def __init__(self, shape, stride, activation, padding='SAME', stddev=None, bias=None):
        self.shape = shape
        self.stride = stride
        self.padding = padding
        if stddev is not None:
            initW = initWeight(self.shape, stddev)
        else:
            initW = initWeight(self.shape)
        if bias is not None:
            initB = initBias(self.shape[-1:], bias)
        else:
            initB = initBias(self.shape[-1:])
            
        self.params = {}
        self.params['W'] = tf.Variable(initW)
        self.params['b'] = tf.Variable(initB)
        self.g = act_list[activation]
    
    def forward(self, X):
        Z = tf.nn.conv2d(X, self.params['W'], strides=[1, self.stride, self.stride, 1], padding=self.padding) + self.params['b']
        return self.g(Z)

pool_type = {'MAX':tf.nn.max_pool, \
             'AVG':tf.nn.avg_pool}

class PoolLayer(object):
    def __init__(self, shape, stride, pooling='AVG', padding='SAME'):
        self.shape = (1, shape[0], shape[1], 1)
        self.stride = (1, stride, stride, 1)
        self.pooltype = pooling
        self.padding = padding
        self.pool = pool_type[self.pooltype]
    
    def forward(self, X):
        return self.pool(X, self.shape, self.stride, self.padding)

class FullyConnectedLayer(object):
    def __init__(self, shape, activation='identity', shared_W=None, stddev=None, bias=None):
        self.shape = shape
        self.params = {}
        if shared_W is None:
            if stddev is not None:
                initW = initWeight(self.shape, stddev)
            else:
                initW = initWeight(self.shape)
            self.params['W'] = tf.Variable(initW)
        else:
            self.params['W'] = shared_W

        if bias is not None:
            initB = initBias(self.shape[-1:], bias)
        else:
            initB = initBias(self.shape[-1:])
        self.params['b'] = tf.Variable(initB)
        self.g = act_list[activation]
    
    def forward(self, X):
        Z = tf.matmul(X, self.params['W']) + self.params['b']
        return self.g(Z)


class FeedForwardNet():   
    n_layer = None
    layers = None
    def __init__(self):
        self.n_layer = 0
        self.layers = []
    
    def addLayer(self, new_layer):
        self.layers += [new_layer]
        self.n_layer += 1
 
    def forward(self, X):
        Xin = X
        for layer in self.layers:
            Xin = layer.forward(Xin)
        return Xin
