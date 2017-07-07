#coding:utf-8

import numpy as np
from utils import dataLoader
import os
import tensorflow as tf
from tensorflow.contrib import rnn


class LstmModel():
    def __init__(self,config):
        #config
        batch_size = config.batch_size
        seq_length = config.seq_length
        rnn_size = config.rnn_size
        num_step = config.seq_length
        class_num = config.class_num
        self.keep_prob = tf.placeholder(tf.float32)
        embed_dim = config.embed_dim

        #feed
        self.input_data = tf.placeholder(tf.int32,[batch_size,seq_length])
        self.targets = tf.placeholder(tf.int64,[batch_size,])
        self.mask = tf.placeholder(tf.float32,[num_step,None])
        self.lr = tf.Variable(0.0,trainable=False)

        #wordembedding
        with tf.variable_scope('word_embedding'):
            embedding = tf.get_variable('embedding',[config.vocab_size,embed_dim])
            inputs = tf.nn.embedding_lookup(embedding,self.input_data)

        #lstm_layer
        lstm_cell = rnn.BasicLSTMCell(rnn_size,forget_bias=0.0,state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
        inputs = tf.nn.dropout(inputs,self.keep_prob)
        cell = rnn.MultiRNNCell([lstm_cell]*config.layer_num,state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size,dtype=tf.float32)

        output =[]
        state = self._initial_state
        with tf.variable_scope('lstm_layer'):
            for time_step in range(num_step):
                if time_step>0:tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(inputs[:,time_step,:],state)
                output.append(cell_output)
        output = output*self.mask[:,:,None]

        with tf.variable_scope('mean_pooling'):
            output = tf.reduce_sum(output,0)/(tf.reduce_sum(self.mask,0)[:,None])

        with tf.variable_scope('fc_layer'):
            fc_w = tf.get_variable('fc_w',[rnn_size,class_num])
            fc_b = tf.get_variable('fc_b',[class_num])
            self.logits = tf.matmul(output,fc_w)+fc_b

        self.loss =tf.nn.sparse_softmax_cross_entropy_with_logits(\
                logits=self.logits,labels=self.targets)
        self.cost = tf.reduce_mean(self.loss)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.pre = tf.argmax(self.logits,1)
        correct_prediction = tf.equal(tf.argmax(self.logits,1),self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #summary
        loss_summary = tf.summary.scalar("loss",self.cost)
        accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

        #learning_rate
        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)
    def assign_new_lr(self,session,lr_value):
            session.run(self._lr_update,feed_dict={self.new_lr:lr_value})

    def batch_normal(self,x,scop ='bn'):
        with tf.variable_scope('bn'):
            x_shape = x.get_shape()
            param_shape = x_shape[-1:]
            axis = list(range(len(x_shape)-1))
            beta = tf.Variable(tf.constant(0.0,shape=param_shape),name='beta',trainable=True)
            gamma = tf.Variable(tf.constant(1.0,shape=param_shape),name='gamma',trainable=True)
            mean = tf.Variable(tf.constant(0,shape=param_shape),name='mean',trainable=False)
            variance = tf.Variable(tf.constant(1,shape=param_shape),name='variance',trainable=False)
            mean,variance = tf.nn.moments(x,axis)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(\
                        x, mean, variance, self.beta, self.gamma,self.epsilon)

    def get_assigner(self,mean,variance):
        return self.ewma_traner.apply([mean,variance])


    def _get_variables(self,name,shape,dtype,weight_decay=0.0,trainable=True):
        if weight_decay>0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        return tf.get_variable(name,\
                shape=shape,dtype=dtype,regularizer=regularizer,trainable=trainable)



