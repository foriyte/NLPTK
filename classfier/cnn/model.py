#coding:utf-8

import numpy as np
from utils import dataLoader
import os
import tensorflow as tf


class CnnModel():
    def __init__(self,config,training=True):
        #config
        batch_size = config.batch_size
        seq_length = config.seq_length
        filter_list = config.filter_list
        num_step = config.seq_length
        class_num = config.class_num
        keep_prob = config.keep_prob
        embed_dim = config.embed_dim
        filter_num = config.filter_num

        #feed
        self.input_data = tf.placeholder(tf.int32,[batch_size,seq_length])
        self.targets = tf.placeholder(tf.int64,[batch_size,])
        self.mask = tf.placeholder(tf.float32,[num_step,None])
        self.lr = tf.Variable(0.0,trainable=False)

        #wordembedding
        with tf.variable_scope('word_embedding'):
            embedding = tf.get_variable('embedding',[config.vocab_size,embed_dim])
            inputs = tf.nn.embedding_lookup(embedding,self.input_data)
            inputs = tf.expand_dims(inputs,-1)

        #conv
        outputs = []
        for i,filter_size in enumerate(filter_list):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embed_dim, 1,filter_num]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                conv = tf.nn.conv2d(inputs,W,strides=[1, 1, 1, 1],\
                        padding="VALID",name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(h,ksize=[1,seq_length-filter_size+1,1,1],\
                        strides=[1,1,1,1],padding='VALID',name="pool")
                outputs.append(pooled)

	# Combine all the pooled features
        num_filters_total = filter_num * len(filter_list)
        self.h_pool = tf.concat(outputs,3)
        output = tf.reshape(self.h_pool, [-1, num_filters_total])

        if keep_prob<1:
            output = tf.nn.dropout(output,keep_prob)

        with tf.variable_scope('fc_layer'):
            fc_w = tf.get_variable('fc_w',[num_filters_total,class_num])
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


