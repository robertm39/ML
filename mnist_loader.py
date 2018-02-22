# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 08:49:48 2018

@author: rober
"""

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def get_mnist(ls=0.9):
    mnist=input_data.read_data_sets('MNIST_data\\', one_hot=True)
    
    trX, trY, tvX, tvY, teX, teY = mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, mnist.test.images, mnist.test.labels
    
    #Combine training and validation
    trX = trX.tolist()
    trY = trY.tolist()
    
    tvX = tvX.tolist()
    tvY = tvY.tolist()
    
    trX.extend(tvX)
    trY.extend(tvY)
    
    trX = np.asarray(trX)
    trY = np.asarray(trY)
    
    #Uniformly smooth training labels
    high = 0.9 
    b = (1-high)/9
    trY *= (high - b)
    trY += b
    
    return trX, trY, teX, teY