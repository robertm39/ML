# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:47:31 2018

@author: rober
"""

import tensorflow as tf

def noised(inputs, rate, training):
    return tf.layers.dropout(inputs=inputs, rate=rate, training=training)

def uniform_noised(inputs, training, width):
    minval = -width / 2
    maxval = width / 2
    noise = tf.random_uniform(shape=inputs.shape[1:],
                              minval=minval,
                              maxval=maxval)
    noise = tf.expand_dims(noise, axis=0)
    pre_shape = noise.shape
    noise = tf.multiply(noise, tf.cast(training, dtype=tf.float32))
    noise.set_shape(pre_shape) #recover shape information lost from multiplying by training
#    print('noise.shape:', noise.shape)
    return inputs + noise

def conv_and_pool(inputs, filters):
    conv = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=[5,5],
                              kernel_initializer=tf.orthogonal_initializer,
                              use_bias=True,
                              padding='same',
                              activation=tf.nn.crelu)
    pool = tf.layers.average_pooling2d(inputs=conv, pool_size=[2,2], strides=2)
    pool = tf.multiply(2.0, pool)
    return pool

def dense(inputs, units):
    dense = tf.layers.dense(inputs=inputs,
                            units=units,
                            kernel_initializer=tf.orthogonal_initializer,
                            activation=tf.nn.crelu)
    return tf.multiply(2.0, dense)

def get_model(X, training, batch_size):
    #Reshape input into 28x28 grayscale images
#    dropout_rate = 0.4
    
    input_layer = tf.reshape(X, [batch_size, 28, 28, 1])
#    input_layer = uniform_noised(input_layer,
#                                 training=training,
#                                 width=0.2)
#    print(input_layer.shape)
#    input_layer = noised(input_layer, rate=dropout_rate, training=training)
    
    #Convolutional part
    
    #Layer one
    c1_filters = 16
    pool_1 = conv_and_pool(input_layer, c1_filters)
#    pool_1 = noised(pool_1, rate=dropout_rate, training=training)
    
    #Layer two
    c2_filters = 16
    pool_2 = conv_and_pool(pool_1, c2_filters)
#    pool_2 = noised(pool_2, rate=dropout_rate, training=training)
    
    #Dense layers
    flattened = tf.reshape(pool_2, [batch_size, 7*7*c2_filters*2])
    
    dense_1_units = 256
    dense_1 = dense(flattened, dense_1_units)
#    dense_1 = tf.layers.dense(inputs=flattened,
#                              units=dense_1_units,
#                              use_bias=True,
#                              kernel_initializer=tf.orthogonal_initializer,
#                              activation=tf.nn.crelu,
#                              name='dense_1')
#    dense_1 = tf.multiply(dense_1, 2.0)
#    dense_1 = noised(dense_1, rate=dropout_rate, training=training)
    
    dense_2_units = 128
    dense_2 = dense(dense_1, dense_2_units)
#    dense_2 = tf.layers.dense(inputs=dense_1,
#                              units=dense_2_units,
#                              use_bias=True,
#                              kernel_initializer=tf.orthogonal_initializer,
#                              activation=tf.nn.crelu,
#                              name='dense_2')
#    dense_2 = tf.multiply(dense_2, 2.0)
#    dense_2 = noised(dense_2, rate=dropout_rate, training=training)
    
    #Logits
    logits = tf.layers.dense(inputs=dense_2,
                               units=10,
                               use_bias=True,
                               kernel_initializer=tf.orthogonal_initializer,
                               name='logits')
    
    sess = tf.Session()
    return logits, 0.0, sess, 'mnist_deep_ortho2'