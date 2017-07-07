#coding:utf-8
import tensorflow as tf
import numpy as np
import os


class AutoEncoder():

    def __init__(self,config):
        # Parameters
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.n_hidden_1 = config.n_hidden_1
        self.n_hidden_2 = config.n_hidden_2
        self.n_input = config.n_input
        self.keep_prob = config.keep_prob
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_2])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
        }

    def build(self):
        self.X = tf.placeholder("float", [None,self.n_input])
        if  self.keep_prob:
            self.X = tf.nn.dropout(self.X,self.keep_prob)
        encoder_op = self.encoder(self.X)
        decoder_op = self.decoder(encoder_op)
        y_pred = decoder_op
        y_true = tf.nn.sigmoid(self.X)
        y_pred = tf.nn.sigmoid(y_pred)
        #self.cost = tf.nn.sigmoid_cross_entropy_with_logits(\
        #        logits=y_pred,labels=y_true)
        #self.cost = -tf.reduce_sum(y_true*tf.log(y_pred))
        #self.cost = tf.reduce_mean(self.cost)
        self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        loss_summary = tf.summary.scalar("loss",self.cost)

        #mid = tf.nn.relu(tf.add(tf.matmul(\
        #        self.X, self.weights['encoder_h1']),self.biases['encoder_b1']))
        self.feature = tf.nn.sigmoid(tf.add(tf.matmul(\
                self.X,self.weights['encoder_h1']),self.biases['encoder_b1']))


    def encoder(self,x):
        layer_1 = tf.nn.sigmoid(tf.add(\
                tf.matmul(x,self.weights['encoder_h1']),self.biases['encoder_b1']))
        #layer_2 = tf.nn.relu(tf.add(\
        #        tf.matmul(layer_1,self.weights['encoder_h2']),self.biases['encoder_b2']))
        return layer_1

    def decoder(self,x):
        #layer_1 = tf.nn.sigmoid(tf.add(\
        #       tf.matmul(x,self.weights['decoder_h1']),self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(\
                tf.matmul(x,self.weights['decoder_h2']),self.biases['decoder_b2']))
        return layer_2












