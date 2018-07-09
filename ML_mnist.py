#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:40:24 2017

@author: JC
"""


from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess= tf.InteractiveSession()

X=tf.placeholder(tf.float32,shape=[None,784])
Y=tf.placeholder(tf.float32,shape=[None,10])


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

W_fc1 = weight_variable([784,200])
b_fc1 = bias_variable([200])
v_fc1 = tf.nn.sigmoid(tf.matmul(X,W_fc1)+b_fc1)

W_fc2 = weight_variable([200,10])
b_fc2 = bias_variable([10])
v_fc2 = tf.nn.sigmoid(tf.matmul(v_fc1,W_fc2)+b_fc2)

predicted_Y = v_fc2
sess.run(tf.global_variables_initializer())
mse=tf.losses.mean_squared_error(Y,predicted_Y)
train_step = tf.train.GradientDescentOptimizer(0.7).minimize(mse)

for i in range(30000):
    batch = mnist.train.next_batch(200)
    if i % 200 ==0:
        correct_prediction = tf.equal(tf.argmax(predicted_Y,1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print(i,accuracy.eval(feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
    train_step.run(feed_dict={X:batch[0],Y:batch[1]})
    
correct_prediction = tf.equal(tf.argmax(predicted_Y,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval(feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
