#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import scipy.spatial.distance as sd
import os


minsse = 8
max_level = 7
hashword = {}
min_cluster_members = 50
data_file='senvec.npy'
hash_file = 'hash_w2v.txt'
data_dir = 'data'
origin_file = 'origin.txt'

class Cluster():

    def __init__(self,data,hashlist,level,data_dir,filename):
        self.data = data
        self.hashlist = hashlist
        self.level = level
        self.data_dir = data_dir
        self.filename = filename

def kmeans(cluster):
    print 'start %d'%cluster.level
    if len(cluster.data)<min_cluster_members:
        return
    km = KMeans(init='k-means++', n_clusters=2,max_iter=500).fit(cluster.data)
    centerA = km.cluster_centers_[0]
    centerB = km.cluster_centers_[1]
    labels = km.labels_
    clusterA = []
    clusterB = []
    hashA = []
    hashB = []
    for i in range(len(labels)):
        if labels[i] == 0:
            clusterA.append(cluster.data[i])
            hashA.append(cluster.hashlist[i])
        else:
            clusterB.append(cluster.data[i])
            hashB.append(cluster.hashlist[i])
    clusterA = np.array(clusterA)
    clusterB = np.array(clusterB)
    sseA = computeSse(clusterA,centerA)
    sseB = computeSse(clusterB,centerB)

    if not os.path.isdir(cluster.data_dir):
        os.makedirs(cluster.data_dir)
    hashDA = pd.DataFrame(hashA)
    hashDB = pd.DataFrame(hashB)
    hashDA.to_csv(os.path.join(cluster.data_dir,cluster.filename+'0.txt'),index=False,header=False)
    hashDB.to_csv(os.path.join(cluster.data_dir,cluster.filename+'1.txt'),index=False,header=False)
    np.save(os.path.join(cluster.data_dir,cluster.filename+'0.npy'),clusterA)
    np.save(os.path.join(cluster.data_dir,cluster.filename+'1.npy'),clusterB)
    del cluster.data,cluster.hashlist
    if sseA>minsse and cluster.level<max_level:
        newclusterA = Cluster(clusterA,hashA,cluster.level+1,cluster.data_dir+'/cluster0',\
                cluster.filename+'0')
        kmeans(newclusterA)
    if sseB>minsse and cluster.level<max_level:
        newclusterB = Cluster(clusterB,hashB,cluster.level,cluster.data_dir+'/cluster1',\
                cluster.filename+'1')
        kmeans(newclusterB)


def computeSse(data,center):
    return sd.cdist([center],data)[0].mean()


def filterAll(cluster_dir):
    print 'start....'
    #载入字典
    with open(os.path.join(data_dir,origin_file),'r') as rf:
        for line in rf.readlines():
            blocks = line.strip().split('\t')
            hashword[blocks[0]]=blocks[1]
    savetitles(cluster_dir,'cluster','title')

def savetitles(cluster_dir,filename,titlename):
    hasha = pd.read_csv(os.path.join(cluster_dir,filename+'0.txt'),\
            names=['hash'])['hash'].values
    hashb = pd.read_csv(os.path.join(cluster_dir,filename+'1.txt'),\
            names=['hash'])['hash'].values
    with open(os.path.join(cluster_dir,titlename+'0.txt'),'wb') as wf:
        for vid in hasha:
            wf.write(hashword[vid]+'\n')
    with open(os.path.join(cluster_dir,titlename+'1.txt'),'wb') as wf:
        for vid in hashb:
            wf.write(hashword[vid]+'\n')
    cluster_dirA = os.path.join(cluster_dir,'cluster0')
    if os.path.exists(cluster_dirA):
        savetitles(cluster_dirA,filename+'0',titlename+'0')
    cluster_dirB = os.path.join(cluster_dir,'cluster1')
    if os.path.exists(cluster_dirB):
        savetitles(cluster_dirB,filename+'1',titlename+'1')


if __name__=='__main__':

    #hashid = pd.read_csv(os.path.join(data_dir,hash_file),names=['id'])['id'].values
    #data = np.load(os.path.join(data_dir,data_file))
    #root = Cluster(data,hashid,0,'bw2vcluster','cluster')
    #kmeans(root)
    filterAll('bw2vcluster')

