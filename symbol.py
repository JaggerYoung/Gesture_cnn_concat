import find_mxnet
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import sys

def get_cnn(input_data):
    # stage 1
    conv1 = mx.symbol.Convolution(data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=2048)
    return fc1

def cnn_concat(seq_len, num_hidden, num_label):
    input_data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    wordvec = mx.symbol.SliceChannel(data=input_data, num_outputs=seq_len, axis=2)   
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = get_cnn(wordvec[seqidx])
        hidden_all.append(hidden)
    hidden_concat = mx.symbol.Concat(*hidden_all, dim=1)
    
    fc1 = mx.symbol.FullyConnected(data=hidden_concat, num_hidden=num_hidden)
    act1 = mx.symbol.Activation(data=fc1, act_type='relu')
    pred = mx.symbol.FullyConnected(data=act1, num_hidden=num_label, name='fc')

    sm = mx.symbol.SoftmaxOutput(data=pred, label=label, name='softmax')
    return sm

