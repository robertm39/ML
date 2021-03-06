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

def conv_and_pool(inputs, filters, layer):
    av_inputs = tf.reduce_mean(inputs, axis=[0, 1, 2], keep_dims=True)
    conv = tf.layers.conv2d(inputs=inputs - av_inputs,
                              filters=filters,
                              kernel_size=[5,5],
                              kernel_initializer=tf.orthogonal_initializer,
                              use_bias=True,
                              padding='same',
                              activation=tf.nn.crelu,
                              name='conv'+str(layer))
    pool = tf.layers.average_pooling2d(inputs=conv, pool_size=[2,2], strides=2)
    pool = tf.multiply(2.0, pool)
    return pool

def dense(inputs, units, layer):
#    dense = tf.layers.dense(inputs=inputs,
#                            units=units,
#                            kernel_initializer=tf.orthogonal_initializer,
#                            activation=tf.nn.crelu)
#    
#    normalized = inputs - av_inputs
#    norm_mul = tf.matmul(normalized, weights)
#    mean_mul = tf.matmul(av_inputs, weights)
#    st_mean_mul = tf.stop_gradient(mean_mul) #Don't take into account the effect if input means
#    
#    dense = norm_mul + st_mean_mul #The output is the same as normal dense, but the gradient is different
    
    init = tf.orthogonal_initializer()
    weights = tf.Variable(init.__call__(shape=[int(inputs.shape[1]), units]), trainable=True, name='dense_kernel_' + str(layer))
    bias = tf.Variable(tf.zeros(shape=[units]), trainable=True, name='dense_bias_' + str(layer))
    
    av_inputs = tf.reduce_mean(inputs, axis=0, keep_dims=True)
    dense = tf.matmul(inputs - av_inputs, weights)
#    mean_mul = tf.matmul(av_inputs, weights)
    
#    st_mean_mul = tf.stop_gradient(mean_mul)
    
#    dense = dense - mean_mul + st_mean_mul#Gradient acts like normalized
    
#    missing_grads = tf.gradients(mean_mul, weights)
#    missing_grads = tf.stack(missing_grads, axis=1)
#    missing_grads = missing_grads[0, 0, :]
#    print(missing_grads.shape)
#    missing_grads = tf.stop_gradient(missing_grads)
#    missing_grads = tf.reduce_mean(missing_grads)
#    print(missing_grads.shape)
#    print('')
#    #Add gradient of missing grads to bias
#    bias = bias + bias*missing_grads - tf.stop_gradient(bias*missing_grads)
##    bias = bias + bias*st_mean_mul - tf.stop_gradient(bias*st_mean_mul)
    
    #Speed up bias training to make up for untransferred gradient
    dense = dense + bias
    
    dense = tf.nn.crelu(dense)
    
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
    return logits, 0.0, sess, 'mnist_dpor_bcorrect'