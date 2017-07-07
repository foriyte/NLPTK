#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import pandas as pd
import os
import collections


class dataLoader():
    def __init__(self,batch_size,vocab_size,data_dir,seq_length,vali_rate,trainable=True):
        self.input_file = os.path.join(data_dir,'input_words.txt')
        self.label_file = os.path.join(data_dir,'label.txt')
        self.vocab_file = os.path.join(data_dir,'vocab.txt')
        self.traindata_file = os.path.join(data_dir,'traindata.npy')
        self.trainlabel_file = os.path.join(data_dir,'trainlabel.npy')
        self.trainmask_file = os.path.join(data_dir,'trainmask.npy')
        self.validata_file = os.path.join(data_dir,'validata.npy')
        self.valilabel_file = os.path.join(data_dir,'valilabel.npy')
        self.valimask_file = os.path.join(data_dir,'valimask.npy')
        self.sample_file = os.path.join(data_dir,'sample.txt')
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.vali_rate = vali_rate
        if trainable:
            if not (os.path.exists(self.valimask_file)):
                print 'preprocess data....'
                self.preprocess()
            else:
                print 'loaddata....'
                self.loaddata()
            self.init_batch()
        else:
            self.sample()

    def preprocess(self):
        self.createVocab()
        self.data = self.mapwordstoid(self.data)
        self.splitvaliset()
        self.sortSentence()
        self.pading_set()
        self.savedata()

    def loaddata(self):
        self.train_data = np.load(self.traindata_file)
        self.train_label = np.load(self.trainlabel_file)
        self.train_mask = np.load(self.trainmask_file)
        self.vali_data = np.load(self.validata_file)
        self.vali_label = np.load(self.valilabel_file)
        self.vali_mask = np.load(self.valimask_file)
        self.createVocabList()


    def savedata(self):
        np.save(self.traindata_file,self.train_data)
        np.save(self.trainlabel_file,self.train_label)
        np.save(self.trainmask_file,self.train_mask)
        np.save(self.validata_file,self.vali_data)
        np.save(self.valilabel_file,self.vali_label)
        np.save(self.valimask_file,self.vali_mask)


    def pading_set(self,trainable = True):
        def pading_(X,Y,type='post'):
            new_x = np.zeros([len(X),self.seq_length],dtype=np.int32)
            new_y = np.zeros(len(X),dtype=np.int32)
            mask = np.zeros([self.seq_length,len(X)],dtype=np.int32)
            for i,(x,y) in enumerate(zip(X,Y)):
                if len(x)<=self.seq_length:
                    if type=='pre':
                        new_x[i,:len(x)] = x
                        mask[:len(x),i] = 1
                    if type=='post':
                        new_x[i,(self.seq_length-len(x)):] = x
                        mask[(self.seq_length-len(x)):,i] = 1
                    new_y[i] = y
                else:
                    if type=='pre':
                        new_x[i] = x[:self.seq_length]
                    if type=='post':
                        new_x[i] = x[(len(x)-self.seq_length):]
                    mask[:,i] = 1
                    new_y[i] = y
            if trainable:
                return new_x,new_y,mask
            else:
                return new_x,mask
        if trainable:
            self.train_data,self.train_label,self.train_mask = pading_(self.train_data,self.train_label)
            self.vali_data,self.vali_label,self.vali_mask = pading_(self.vali_data,self.vali_label)
        else:
            self.sample_data,self.sample_mask = \
                    pading_(self.sample_data,np.zeros(len(self.sample_data)))


    def sortSentence(self):
        def len_argsort(seq):
            return sorted(range(len(seq)),key=lambda x:len(seq[x]))
        sorted_id = len_argsort(self.train_data)
        self.train_data = [self.train_data[i] for i in sorted_id]
        self.train_label = [self.train_label[i] for i in sorted_id]

        sorted_id = len_argsort(self.vali_data)
        self.vali_data = [self.vali_data[i] for i in sorted_id]
        self.vali_label = [self.vali_label[i] for i in sorted_id]

    def mapwordstoid(self,origindata):
        data = []
        for line in origindata:
            words = line.strip().split()
            vec = []
            for w in words:
                if w in self.vocab.keys():
                    vec.append(self.vocab[w])
            data.append(vec)
        return data

    def splitvaliset(self):
        self.data = np.array(self.data)
        self.label = pd.read_table(self.label_file,names=['label'])['label'].values
        length = len(self.data)
        sidx = np.random.permutation(length)
        num_vali = int(np.round(length*self.vali_rate))
        self.train_data = [self.data[i] for i in sidx[num_vali:]]
        self.train_label = [self.label[i] for i in sidx[num_vali:]]
        self.vali_data = [self.data[i] for i in sidx[:num_vali]]
        self.vali_label = [self.label[i] for i in sidx[:num_vali]]
        del self.data,self.label
        def removeEos(x):
            return [[0 if w>self.vocab_size else w for w in sen] for sen in x]
        self.train_data = removeEos(self.train_data)
        self.vali_data = removeEos(self.vali_data)


    def createVocab(self):
        with open(self.input_file,'r') as rf:
            self.data = rf.readlines()
        if not os.path.exists(self.vocab_file):
            wordsDic=collections.Counter()
            for line in self.data:
                words = line.strip().split()
                wordsDic.update(words)
            words = wordsDic.keys()
            freqs = wordsDic.values()
            s = np.argsort(freqs)[::-1]
            self.vocab = collections.OrderedDict()
            for i,index in enumerate(s):
                self.vocab[words[index]] = i
            #self.vocab_size = len(self.vocab)
            with open(self.vocab_file,'wb') as wf:
                for i in self.vocab.keys():
                    wf.write(i+'\n')
        else:
            self.createVocabList()

    def createVocabList(self):
        with open(self.vocab_file,'r') as f:
            self.vocab = collections.OrderedDict()
            for i,line in enumerate(f.readlines()):
                self.vocab[line.strip()] = i
        #self.vocab_size = len(self.vocab)

    def init_batch(self,trainable = True):
        if trainable:
            self.train_num = int(len(self.train_data)/self.batch_size)
            self.vali_num = int(len(self.vali_data)/self.batch_size)
        else:
            self.sample_num = len(self.sample_data)
            self.batch_size = 1

    def get_batches(self,mode):
        if mode=='train':
            for i in range(self.train_num):
                x = self.train_data[i*self.batch_size:(i+1)*self.batch_size]
                y = self.train_label[i*self.batch_size:(i+1)*self.batch_size]
                mask = self.train_mask[:,i*self.batch_size:(i+1)*self.batch_size]
                yield x,y,mask
        if mode=='validation':
            for i in range(self.vali_num):
                x = self.vali_data[i*self.batch_size:(i+1)*self.batch_size]
                y = self.vali_label[i*self.batch_size:(i+1)*self.batch_size]
                mask = self.vali_mask[:,i*self.batch_size:(i+1)*self.batch_size]
                yield x,y,mask
        if mode=='sample':
            for i in range(self.sample_num):
                x = self.sample_data[i*self.batch_size:(i+1)*self.batch_size]
                mask = self.sample_mask[:,i*self.batch_size:(i+1)*self.batch_size]
                yield x,mask


    def sample(self):
        self.createVocabList()
        lines = open(self.sample_file,'r').readlines()
        self.sample_data  = self.mapwordstoid(lines)
        self.pading_set(False)
        self.init_batch(False)

