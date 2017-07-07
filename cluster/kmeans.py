#coding:utf-8
import numpy as np
import os
from sklearn.cluster import KMeans
import pandas as pd
import scipy.spatial.distance as sd
import random

data_dir = 'data' #数据路径
vecdim = 50
senvec_file = 'senvec.txt'
senvecfile = 'senvec.npy'
hash_file='hash_w2v.txt' #w2v vedioid 列表,和senvec.npy 一一对应
origin_file = 'origin.txt'
hashword = {}

def vecTonpy():
    np.save(os.path.join(data_dir,senvecfile),np.array([w.split() for w in \
            open(os.path.join(data_dir,senvec_file),'r')],dtype=np.float))

def kmeans(n_cluster,data,hashid,cluster_dir,filename):
    assert len(data)==len(hashid),'length error!'
    print 'start kmeans....'
    clusters = KMeans(init='k-means++', n_clusters=n_cluster,max_iter=500).fit_predict(data)
    if not os.path.isdir(cluster_dir):
        os.makedirs(cluster_dir)
    np.save(os.path.join(cluster_dir,'kclusters.npy'),clusters)
    for i in range(n_cluster):
        print 'start %d branchs ....'%i
        hf = open(os.path.join(cluster_dir,filename+'_%d.txt'%i),'wb')
        nf = os.path.join(cluster_dir,filename+'_%d.npy'%i)
        index = []
        for j in xrange(len(hashid)):
            if clusters[j]==i:
                hf.write(hashid[j]+'\n')
                index.append(j)
        hf.close()
        np.save(nf,data[index])

def hierarchykmeans(n_cluster=4,cluster_dir='w2vcluster'):
    data = np.load(os.path.join(data_dir,senvecfile))
    hashid = pd.read_csv(os.path.join(data_dir,hash_file),names=['id'])['id'].values
    kmeans(n_cluster,data,hashid,cluster_dir,'cluster')

    for i in range(n_cluster):
        print 'start %d sub cluster ....'%i
        idata = np.load(os.path.join(cluster_dir,'cluster'+'_%d.npy'%i))
        ihash = pd.read_csv(os.path.join(cluster_dir,'cluster'+'_%d.txt'%i),\
                names=['id'])['id'].values
        kmeans(n_cluster,idata,ihash,os.path.join(cluster_dir,'cluster%d'%i),'cluster_%d'%i)

#提取每个类别的文本,存到相应目录下
def filterAll(n_cluster = 4,cluster_dir = 'w2vcluster'):
    print 'start....'
    #载入字典
    with open(os.path.join(data_dir,origin_file),'r') as rf:
        for line in rf.readlines():
            blocks = line.strip().split('\t')
            hashword[blocks[0]]=blocks[1]
    print 'load hashwordsdic ok..... '
    for i in range(n_cluster):
        subcluster_dir = os.path.join(cluster_dir,'cluster%d'%i)
        print '....cluster %d.....'%i
        for j in range(n_cluster):
            hashfile = os.path.join(subcluster_dir,'cluster_%d_%d.txt'%(i,j))
            savetiles(hashfile,subcluster_dir,(i,j))


def savetiles(hash_file,cluster_dir,index):
    hashid = pd.read_csv(hash_file,names=['hash'])['hash'].values
    with open(os.path.join(cluster_dir,'title_%d_%d.txt'%index),'wb') as wf:
        for i in range(len(hashid)):
            wf.write(hashid[i]+','+hashword[hashid[i]]+'\n')



if __name__=='__main__':
    #vecTonpy()
    #hierarchykmeans()
    filterAll()



