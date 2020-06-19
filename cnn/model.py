# -*- coding: utf-8 -*-

import numpy as np

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def createModel(nbClasses,imageSize):
        print("[+] Creating model...")
        
        convnet = input_data(shape=[None, imageSize, 258, 1], name='input')

        convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 1024, 2, activation='elu', weights_init="Xavier")
        convnet = max_pool_2d(convnet, 2)
        
        convnet = fully_connected(convnet, 2048, activation='elu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, nbClasses, activation='softmax', regularizer='L2')
        convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

        model = tflearn.DNN(convnet)
        print("    Model created!")
        return model
