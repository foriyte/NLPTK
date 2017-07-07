#coding:utf-8

import numpy as np
import os
import gensim
import collections
import scipy.spatial.distance as sd

w2vfilename = 'newmergedata/words.txt' #原始分词文本
w2vmodelname = 'newmergedata/w2vmodel' #模型
wordvec_file = 'newmergedata/wordsvec.txt' #词向量表
w2v_min_wordcount = 5 #去除小于此阈值的词频
word_dim = 50 # 词向量维度


d2vfilename ='mergedata/doc.txt'
d2vmodelname = 'mergedata/d2vModel'
d2v_min_sencount = 1
doc_dim = 20
txtDic=[]

LabeledSentence = gensim.models.doc2vec.LabeledSentence


class Sentences():
    def __init__(self,datafile):
        self.filename = datafile
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

#训练word2vec模型，并生成词向量表
def createWord2vecModel():
    sentences = Sentences(w2vfilename)
    model = gensim.models.Word2Vec(sentences,min_count=w2v_min_wordcount,\
            size = word_dim,workers=4)
    model.save(w2vmodelname)
    vocab = collections.Counter()
    for words in sentences:
        vocab.update(words)
    with open(wordvec_file,'wb') as wf:
        wf.write('%d\n'%len([w for w,v in vocab.items() if v >=w2v_min_wordcount]))
        for w,num in vocab.items():
            if num < w2v_min_wordcount:
                continue
            s = w
            for v in model[w]:
                s += ' %f'%v
            wf.write(s+'\n')

def wordstovec():
    model = gensim.models.Word2Vec.load()
    lines = open('sample/sample.txt','r').readlines()
    vecs = np.zeros([len(lines),100],dtype=np.float32)
    for i in range(len(lines)):
        words = lines[i].split()
        vec = np.zeros(100,dtype=np.float32)
        for w in words:
            try:
                vec += model[w]
            except Exception as e:
                print w,e
        vecs[i] = vec/np.sqrt(sum(vec**2))
    np.save('test.npy',vecs)
    return


class Doc():
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield LabeledSentence(line.split(),['SENT_%s' % uid])

def createDoc2vecModel():
    docs = Doc(d2vfilename)
    model = gensim.models.Doc2Vec(docs,size=doc_dim,min_count=d2v_min_sencount)
    model.save(d2vmodelname)
    print model.docvecs



def doctovec(sen):
    model = gensim.models.Doc2Vec.load(d2vmodelname)
    #for i in model.docvecs:
    #    print i
    global txtDic
    with open('mergedata/doc.txt','r') as rf:
        for line in rf.readlines():
            txtDic.append(line.strip())
    docvecs = np.array(model.docvecs)
    #for i in range(2,2):
    getNeghbor(1550,docvecs)

def getNeghbor(j,data):
    enconding = data[j]
    scores = sd.cdist([enconding],data)[0]
    sorted_ids = np.argsort(scores)
    print 'sentence:'
    print txtDic[j]
    print '\n Nearest neighbors by doc2vec:'
    for i in range(1,11):
        print(" %d. %s \n" %(i,txtDic[sorted_ids[i]]))


if __name__=='__main__':
    createWord2vecModel()
    #wordstovec()
    #createDoc2vecModel()
    #doctovec('nihao')


