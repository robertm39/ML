# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:51:05 2018

@author: rober
"""
import sys

import numpy as np

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

from mnist_dpor_bcorrect import get_model
from mnist_loader import get_mnist

X = tf.placeholder('float', [None, 784], name='X')
Y = tf.placeholder('float', [None, 10], name='Y')

training = tf.placeholder('bool', name='training')
batch_size = tf.placeholder('int32', name='batch_size')

print('Building model')
py_x, extra_cost, sess, path = get_model(X, training, batch_size)
print('Model built')
print('Var shapes:')
p1s = []
p2s = []
p3s = []
for v in tf.trainable_variables():
    p1 = str(np.sum(np.prod(v.get_shape().as_list())))
    p2 = str(v.shape)
    p3 = v.name
    p1s.append(p1)
    p2s.append(p2)
    p3s.append(p3)
mp1 = max([len(p) for p in p1s])
mp2 = max([len(p) for p in p2s])
mp3 = max([len(p) for p in p3s])
for i in range(0, len(p1s)):
    p1, p2, p3 = p1s[i], p2s[i], p3s[i]
    pad=2
    string = ''
    string += p1 + ' '*(mp1 - len(p1) + pad)
    string += p2 + ' '*(mp2 - len(p2) + pad)
    string += p3 + ' '*(mp3 - len(p3) + pad)
    print(string)

print('')
print('Params:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
print('')
trbs = 120

print('Adding cost')
extra_cost_weight = 0.001
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_cost = tf.add(cost, tf.multiply(extra_cost_weight, extra_cost))
    global_step = tf.Variable(0, trainable=False)
    epochs_to_decay = 5
    learning_rate = tf.train.exponential_decay(0.001, global_step, (60000*epochs_to_decay) / trbs, 0.5, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_cost, global_step=global_step)
   
print('Adding accuracy')
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) #count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, 'float')) #cast boolean to float to average
    
print('Making Saver')
saver = tf.train.Saver()
    
min_cost = 0.544805431125 #make this part of the nn module

#Initialize variables
print('Initializing variables')
tf.global_variables_initializer().run(session=sess)
#saver.restore(sess, '/tmp/'+ path + '.ckpt') #restore previous vars

print('Loading MNIST')
trX, trY, teX, teY = get_mnist()
teX_len = len(teX)
print('Training samples:', len(trX))

#Step 12:Train model
print('')
print('Starting training')

#The global step tracks batches, so divide by number of batches per epoch to get the number of epochs
start_step = int(sess.run([global_step])[0] // (len(trX) / trbs))

def run(X, Y, batch_size, training, verbose=True):
    accs = []
    costs = []
    ex_costs = []
    for start, end in zip(range(0, len(trX), trbs), range(trbs, len(trX)+1, trbs)):
        if training:
            acc, cost_val, t_cost_val, t = sess.run([acc_op, cost, train_cost, train_op],
                                        feed_dict={X:trX[start:end],
                                                   Y:trY[start:end],
                                                   batch_size:batch_size,
                                                   training:training})
        else:
            acc, cost_val, t_cost_val = sess.run([acc_op, cost, train_cost],
                                        feed_dict={X:trX[start:end],
                                                   Y:trY[start:end],
                                                   batch_size:batch_size,
                                                   training:training})
    
        accs.append(acc)
        ex_cost_val = t_cost_val - cost_val
        ex_cost_val /= extra_cost_weight
        cost_val -= min_cost
        costs.append(cost_val)
        ex_costs.append(ex_cost_val)
        
        if verbose:
            first_part = 'batch ' + start.__str__() + '-' + end.__str__() + ':'
            first_part = first_part + ' ' * (20 - len(first_part))
            
            middle_part = acc.__str__()
            middle_part = middle_part + '0'*(8-len(middle_part)) + ', '
            
            m2_part = '%.10f' % cost_val
            m2_part = m2_part + ',' + ' '*(14-len(m2_part))
            
            end_part = '%.10f' % ex_cost_val
            end_part = end_part + ' '*10
            
            to_print = first_part + middle_part + m2_part + end_part
            print('\r' + to_print, end='', flush=True)
    print('\r', end='', flush=True)
    sys.stdout.flush()
    return accs, costs, ex_costs

#Pre-run

if start_step == 0:
    accs = [] #Go through eval in 1000-sample batches so my computer doesn't go to sleep
    costs = []
    ex_costs = []
    tebs = 1000
    for start, end in zip(range(0, teX_len, tebs), range(tebs, teX_len + 1, tebs)):
        acc, cost_val, t_cost_val = sess.run([acc_op, cost, train_cost],
                                 feed_dict={X:teX[start:end],
                                            Y:teY[start:end],
                                            batch_size:tebs,
                                            training:False})
        accs.append(acc)
        costs.append(cost_val)
        ex_costs.append((t_cost_val - cost_val) / extra_cost_weight) #ex_cost
    
    acc = np.mean(np.asarray(accs))
    t_cost = np.mean(np.asarray(costs))
    t_ex_cost = np.mean(np.asarray(ex_costs))
    
    print('\r', end='', flush=True)
    first_part = '\r' + 'Pre' + ': ' + '%.4f' % acc + ','
    first_part = 'Pre' + ':' + ' '*(14 - len(first_part)) + '%.4f' % acc + ','
    print(first_part, '%.8f' % t_cost + ', ' + '%.10f' % t_ex_cost, flush=True)

#End pre-run

for i in range(0, 500):
    i += start_step
    tr_accs = []
    tr_costs = []
    tr_ex_costs = []
    for start, end in zip(range(0, len(trX), trbs), range(trbs, len(trX)+1, trbs)):
#            acc, cost_val, t_cost_val = sess.run([acc_op, cost, train_cost],
        acc, cost_val, t_cost_val, t = sess.run([acc_op, cost, train_cost, train_op],
                                    feed_dict={X:trX[start:end],
                                               Y:trY[start:end],
                                               batch_size:trbs,
                                               training:True})
    
        tr_accs.append(acc)
        ex_cost_val = t_cost_val - cost_val
        ex_cost_val /= extra_cost_weight
        cost_val -= min_cost
        tr_costs.append(cost_val)
        tr_ex_costs.append(ex_cost_val)
        
        first_part = 'batch ' + start.__str__() + '-' + end.__str__() + ':'
        first_part = first_part + ' ' * (20 - len(first_part))
        
        middle_part = acc.__str__()
        middle_part = middle_part + '0'*(8-len(middle_part)) + ', '
        
        m2_part = '%.10f' % cost_val
        m2_part = m2_part + ',' + ' '*(14-len(m2_part))
        
        end_part = '%.10f' % ex_cost_val
        end_part = end_part + ' '*10
        
        to_print = first_part + middle_part + m2_part + end_part
        print('\r' + to_print, end='', flush=True)
    print('\r', end='', flush=True)
    sys.stdout.flush() #Comment out?
    
    saver.save(sess, '/tmp/'+ path + '.ckpt')
    
    accs = [] #Go through eval in 1000-sample batches so my computer doesn't go to sleep
    costs = []
    ex_costs = []
    tebs = 1000
    for start, end in zip(range(0, teX_len, tebs), range(tebs, teX_len + 1, tebs)):
        acc, cost_val, t_cost_val = sess.run([acc_op, cost, train_cost],
                                 feed_dict={X:teX[start:end],
                                            Y:teY[start:end],
                                            batch_size:tebs,
                                            training:False})
        accs.append(acc)
        costs.append(cost_val)
        ex_costs.append((t_cost_val - cost_val) /  extra_cost_weight) #ex_cost
    
    tr_acc = np.mean(np.asarray(tr_accs))
    tr_cost = np.mean(np.asarray(tr_costs))
    tr_ex_cost = np.mean(np.asarray(tr_ex_costs))
    
    acc = np.mean(np.asarray(accs))
    t_cost = np.mean(np.asarray(costs))
    t_ex_cost = np.mean(np.asarray(ex_costs))
    
    print('\r', end='', flush=True)
    first_part = '\r' + i.__str__() + ': ' + '%.4f' % acc + ','
    first_part = i.__str__() + ':' + ' '*(14 - len(first_part)) + '%.4f' % acc + ','
    print(first_part, '%.10f' % t_cost + ', ' + '%.10f' % t_ex_cost + ', ' + '%.5f' % tr_acc + ', ' + '%.10f' % tr_cost + ', ' + '%.10f' % tr_ex_cost, flush=True)
