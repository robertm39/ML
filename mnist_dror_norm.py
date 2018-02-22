# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:56:08 2018

@author: rober
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:14:25 2018

@author: rober
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:47:31 2018

@author: rober
"""

import tensorflow as tf

def rms_norm(inputs, axis):
    squ = tf.square(inputs)
    means = tf.reduce_mean(squ, axis=axis, keep_dims=True)
    rms = tf.sqrt(means)
    return inputs / rms

def conv_and_pool(inputs, filters, layer):
    init = tf.orthogonal_initializer()
    weights = tf.Variable(init.__call__(shape=[5, 5, int(inputs.shape[3]), filters]), trainable=True)
    weights = rms_norm(weights, axis=3)
    
    av_inputs = tf.reduce_mean(inputs, axis=[0, 1, 2], keep_dims=True)
    
#    conv = tf.layers.conv2d(inputs=inputs - av_inputs,
#                              filters=filters,
#                              kernel_size=[5,5],
#                              kernel_initializer=tf.orthogonal_initializer,
#                              use_bias=False,
#                              padding='same',
#                              activation=tf.nn.crelu,
#                              name='conv'+str(layer))
    conv = tf.nn.conv2d(inputs - av_inputs,
                        weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name='conv' + str(layer))
    conv = tf.nn.crelu(conv)
    
    pool = tf.layers.average_pooling2d(inputs=conv, pool_size=[2,2], strides=2)
    pool = tf.multiply(2.0, pool)
    return pool

def dense(inputs, units, layer):
#    av_inputs = tf.reduce_mean(inputs, axis=0, keep_dims=True)
#    dense = tf.layers.dense(inputs=inputs - av_inputs,
#                            units=units,
#                            kernel_initializer=tf.orthogonal_initializer,
#                            use_bias=False,
#                            activation=tf.nn.crelu,
#                            name='dense'+str(layer))
    init = tf.orthogonal_initializer()
    weights = tf.Variable(init.__call__(shape=[int(inputs.shape[1]), units]), trainable=True, name='dense_kernel_' + str(layer))
    
    weights = rms_norm(weights, axis=1)
    
    av_inputs = tf.reduce_mean(inputs, axis=0, keep_dims=True)
    dense = tf.matmul(inputs - av_inputs, weights)
    
    return tf.multiply(2.0, dense)

def get_model(X, training, batch_size):
    #Reshape input into 28x28 grayscale images
    input_layer = tf.reshape(X, [batch_size, 28, 28, 1])
    
    #Convolutional part
    
    #Layer one
    c1_filters = 16
    pool_1 = conv_and_pool(input_layer, c1_filters, layer=1)
    
    #Layer two
    c2_filters = 16
    pool_2 = conv_and_pool(pool_1, c2_filters, layer=2)
    
    #Dense layers
    flattened = tf.reshape(pool_2, [batch_size, 7*7*c2_filters*2])
    
    dense_1_units = 256
    dense_1 = dense(flattened, dense_1_units, layer=1)
    
    dense_2_units = 128
    dense_2 = dense(dense_1, dense_2_units, layer=2)
    
    #Logits
    logits = tf.layers.dense(inputs=dense_2,
                               units=10,
                               use_bias=True,
                               kernel_initializer=tf.orthogonal_initializer,
                               name='logits')
    
    sess = tf.Session()
    return logits, 0.0, sess, 'mnist_deep_ortho2'