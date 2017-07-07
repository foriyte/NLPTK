#coding:utf-8
import numpy as np
import scipy.spatial.distance as sd
import pandas as pd
import os

hashDic = {}
txtDic = {}
data = None
hashlist = []
data_dir = 'data'
origin_file = 'origin.txt'
hash_file = 'hash_w2v.txt'
data_file = 'senvec.npy'

def preprocess():
    global txtDic,data,hashlist,hashDic
    with open(os.path.join(data_dir,origin_file),'r') as tf:
        for line in tf.readlines():
            blocks = line.replace('\n','').split('\t')
            txtDic[blocks[0]] = blocks[1]


    hashlist  = pd.read_csv(os.path.join(data_dir,hash_file),names=['hash'])['hash'].values

    for i,item in enumerate(hashlist):
        hashDic[item] = i
    data = np.load(os.path.join(data_dir,data_file))

def getNeghbor(j):
    scores = sd.cdist([data[j]],data)[0]
    sorted_ids = np.argsort(scores)
    print 'sentence:'
    print txtDic[hashlist[j]]
    print '\n Nearest neighbors by word2vec:'
    for i in range(1,11):
        print(" %d. %s \n" %(i,txtDic[hashlist[sorted_ids[i]]]))

#wf = open('samplebyknn.txt','wb')
def writeNeighbor(j):
    scores = sd.cdist([data[j]],data)[0]
    sorted_ids = np.argsort(scores)
    wf.write('\nsentence:\n')
    wf.write(txtDic[hashlist[j]]+'\n')
    wf.write('\n Nearest neighbors by word2vec:\n')
    for i in range(1,11):
        vid = hashlist[sorted_ids[i]]
        wf.write(" %d. %s\n"%(i,txtDic[vid]))


if __name__== '__main__':
    preprocess()
    for i in range(10):
        getNeghbor(i)
