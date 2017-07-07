#coding:utf-8
#util
import os
import cPickle
import numpy as np
import pandas as pd
import scipy as sp
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#ml
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


vecdim = 50
data_dir = 'mergedata'
senvec_file = 'short_senvec.txt'
senvecfile = 'short_senvec.npy'
labelfile = 'labeldic.pkl'
hashfile = 'hash_w2v.txt'

def vecTonpy():
    np.save(os.path.join(data_dir,senvecfile),np.array([w.split() for w in \
            open(os.path.join(data_dir,senvec_file),'r')],dtype=np.float))
hashid = None

def loadLabel():
    global hashid
    with open(os.path.join(data_dir,labelfile),'r') as rf:
        labeldic = cPickle.load(rf)
    hashid = pd.read_csv(os.path.join(data_dir,hashfile),\
            names=['id'])['id'].values
    return np.array([labeldic[str(w)] for w in hashid],dtype=np.int32)

def train():
    global hashid
    feature = np.load(os.path.join(data_dir,senvecfile))
    labels = loadLabel()
    length = len(labels)
    assert len(labels)==len(feature)==len(hashid),'length error!'
    index = np.arange(length,dtype=np.int32)
    np.random.shuffle(index)
    feature = feature[index]
    labels = labels[index]
    hashid = np.array(hashid)
    hashid = hashid[index]
    k_fold = cross_validation.KFold(n=length/6,n_folds = 5)
    i = 1
    for train_indices,test_indices in k_fold:
        print 'start train with %d'%i
        i += 1
        params_rf = {
                'n_estimators':100,
                'max_features':'log2',
                'max_depth':3,
                'n_jobs':-1
                }
        params_knn = {
                'n_neighbors':10,
                'algorithm':'kd_tree',
                'n_jobs':-1
                }
        params_gbdt = {
                'max_depth':7
                }
        #model = LinearRegression()
        #model = GradientBoostingRegressor(**params_gbdt)
        #model = KNeighborsRegressor(**params_knn)
        model = neighbors.KNeighborsClassifier(5,'distance')
        #model = RandomForestClassifier(**params_rf)

        model.fit(feature[train_indices],labels[train_indices])
        prediction = model.predict(feature[test_indices])
        evaluate(prediction,labels[test_indices],hashid[test_indices])


hehe = open('test.txt','wb')
h = collections.Counter()
errorlist = []
def evaluate(pre,lab,testhash):
    lab = lab.T
    for i in range(len(lab)):
        if pre[i]!=lab[i]:
            h[lab[i]] = h.get(lab[i],0)+1
            errorlist.append(testhash[i])
        hehe.write('%d  %d\n'%(pre[i],lab[i]))
    print 'acc:',accuracy_score(pre,lab)

if __name__=='__main__':
    vecTonpy()
    #train()
    #print h
    #np.save('test.npy',np.array(errorlist))
