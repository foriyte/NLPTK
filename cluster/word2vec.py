#coding:utf-8

import numpy as np
import os
import gensim
import collections

filename = 'data/words.txt' #原始分词文本
modelname = 'data/w2vModel' #模型
wordvec_file = 'data/wordsvec.txt' #词向量表
min_wordcount = 4 #去除小于此阈值的词频
word_dim = 50 # 词向量维度

class Sentences():
    def __init__(self,datafile):
        self.filename = datafile
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

#训练word2vec模型，并生成词向量表
def createWord2vecModel():
    sentences = Sentences(filename)
    model = gensim.models.Word2Vec(sentences,min_count=min_wordcount,\
            size = word_dim,workers=4)
    model.save(modelname)
    vocab = collections.Counter()
    for words in sentences:
        vocab.update(words)
    with open(wordvec_file,'wb') as wf:
        wf.write('%d\n'%len([w for w,v in vocab.items() if v >=min_wordcount]))
        for w,num in vocab.items():
            if num < min_wordcount:
                continue
            s = w
            for v in model[w]:
                s += ' %f'%v
            wf.write(s+'\n')




if __name__=='__main__':
    createWord2vecModel()



