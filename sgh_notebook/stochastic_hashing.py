
import time
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import function

import scipy.io as sio
import sys

"""This is a simple demonstration of the stochastic generative hashing algorithm 
with linear decoder and encoder on MNIST dataset. 

Created by Bo Dai 2016"""


def VAE_stoc_neuron(alpha, dim_input, dim_hidden, batch_size, learning_rate, max_iter, xtrain, xvar, xmean):
    
    g = tf.Graph()
    dtype = tf.float32
    
    with g.as_default():
        x = tf.placeholder(dtype, [None, dim_input])
        
        # define doubly stochastic neuron with gradient by DeFun
        @function.Defun(dtype, dtype, dtype, dtype)
        def DoublySNGrad(logits, epsilon, dprev, dpout):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            # {-1, 1} coding
            # yout = tf.sign(prob - epsilon)

            # biased
            dlogits = prob * (1 - prob) * (dprev + dpout)
                        
            depsilon = dprev
            return dlogits, depsilon

        @function.Defun(dtype, dtype, grad_func=DoublySNGrad)
        def DoublySN(logits, epsilon):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            return yout, prob
        
        with tf.name_scope('encode'):
            wencode = tf.Variable(tf.random_normal([dim_input, dim_hidden], stddev=1.0 / tf.sqrt(float(dim_input)), dtype=dtype),
                                       name='wencode')
            bencode = tf.Variable(tf.random_normal([dim_hidden], dtype=dtype), name='bencode')
            hencode = tf.matmul(x, wencode) + bencode
            # determinastic output
            hepsilon = tf.ones(shape=tf.shape(hencode), dtype=dtype) * .5
            
        yout, pout = DoublySN(hencode, hepsilon)
        
        with tf.name_scope('decode'):
            wdecode = tf.Variable(tf.random_normal([dim_hidden, dim_input], stddev=1.0 / tf.sqrt(float(dim_hidden)), dtype=dtype), 
                                  name='wdecode')
        with tf.name_scope('scale'):
            scale_para = tf.Variable(tf.constant(xvar, dtype=dtype), name="scale_para")
            shift_para = tf.Variable(tf.constant(xmean, dtype=dtype), name="shift_para")
            
        xout = tf.matmul(yout, wdecode) * tf.abs(scale_para) + shift_para
        
        monitor = tf.nn.l2_loss(xout - x, name=None) 
        # loss = monitor + alpha * tf.reduce_sum(tf.reduce_sum(yout * tf.log(pout) + (1 - yout) * tf.log(1 - pout))) + beta * tf.nn.l2_loss(wdecode, name=None)
        loss = monitor + alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(hencode, yout)) + beta * tf.nn.l2_loss(wdecode, name=None) + beta * tf.nn.l2_loss(wencode, name=None)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

        sess = tf.Session(graph=g)
        sess.run(tf.initialize_all_variables())
        
        train_err = []
        for i in xrange(max_iter):
            indx = np.random.choice(xtrain.shape[0], batch_size)
            xbatch = xtrain[indx]
            _, monitor_value, loss_value = sess.run([train_op, monitor, loss], feed_dict={x: xbatch})

            if i % 100 == 0:
                print('Num iteration: %d Loss: %0.04f Monitor Loss %0.04f' %(i, loss_value / batch_size, monitor_value / batch_size))
                train_err.append(loss_value)
   

            if i % 2000 == 0:
                learning_rate = 0.5 * learning_rate


        node_list = ['yout', 'pout', 'xout', 'wencode', 'bencode', 'wdecode', 'scale_para', 'shift_para']
        t_vars = tf.trainable_variables()

        para_list = {}
        for var in t_vars:
            para_list[var.name] = sess.run(var)
    
    return g, node_list, para_list, train_err


if __name__ == "__main__":
    # prepare data
    # please replace the dataset with your own directory.
    traindata = sio.loadmat('../dataset/mnist_training.mat')
    testdata = sio.loadmat('../dataset/mnist_test.mat')
    xtrain = traindata['Xtraining']
    xtest = testdata['Xtest']

    xmean = xtrain.mean(axis=0).astype('float64')
    xvar = np.clip(xtrain.var(axis=0), 1e-7, np.inf).astype('float64')

    # algorithm parameters
    dim_input = 28 * 28
    # length of bits
    dim_hidden= int(sys.argv[1]) 
    print('dim of hidden variable is %d' %(dim_hidden))

    batch_size = 500
    learning_rate = 1e-2
    max_iter = 5000
    alpha = 1e-3
    beta = 1e-3

    # start training
    start_time = time.time()
    g, node_list, para_list, train_err = VAE_stoc_neuron(alpha, dim_input, dim_hidden, batch_size, learning_rate, max_iter, xtrain, xvar, xmean)
    end_time = (time.time() - start_time)

    print('Running time: %0.04f s' %end_time)

    W = para_list['encode/wencode:0']
    b = para_list['encode/bencode:0']
    U = para_list['decode/wdecode:0']
    shift = para_list['scale/shift_para:0']
    scale = para_list['scale/scale_para:0']

    # encoding
    logits = np.dot(np.array(xtest), W) + b
    # epsilon = np.random.uniform(0, 1, logits.shape)
    epsilon = 0.5 

    pres = 1.0 / (1 + np.exp(-logits))
    hres = (np.sign(pres - epsilon) + 1.0) / 2.0

    trainlogits = np.dot(np.array(xtrain), W) + b
    # epsilon = np.random.uniform(0, 1, logits.shape)

    trainpres = 1.0 / (1 + np.exp(-trainlogits))
    htrain = (np.sign(trainpres - epsilon) + 1.0) / 2.0
    htest = hres

    filename = 'SGH_mnist_'+ str(dim_hidden) + 'bit.mat'
    sio.savemat(filename, {'htrain':htrain, 'htest': hres, 'traintime': end_time, 'W': W, 'b': b, 'U': U, 'shift':shift, 'scale':scale})
