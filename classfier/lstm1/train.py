#coding:utf-8
import numpy as np
import tensorflow as tf
from model import LstmModel
from utils import dataLoader
import time
from time import sleep
import os
import cPickle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.01,'the learning rate')
flags.DEFINE_float('lr_decay',0.5,'the learning rate decay')
flags.DEFINE_integer('vocabulary_size',3500,'vocabulary_size')
flags.DEFINE_integer('embedding_dim',128,'embedding dim')
flags.DEFINE_integer('hidden_neural_size',128,'LSTM hidden neural size')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
flags.DEFINE_string('data_path','data','dataset path')
flags.DEFINE_integer('seq_length',50,'max_len of training sentence')
flags.DEFINE_integer('vali_rate',0.2,'rate of validation')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('class_num',7,'class num')
flags.DEFINE_float('keep_prob',1,'dropout rate')
flags.DEFINE_integer('num_epoch',20,'num epoch')
flags.DEFINE_integer('check_point_every',100,'checkpoint every num epoch ')
flags.DEFINE_string('init_from_model','save','last model path')
flags.DEFINE_string('save','save','model save path')
flags.DEFINE_string('log','log','log path')
flags.DEFINE_integer('max_decay_epoch',1,'num epoch')

kemu = ['物理','化学','生物','地理','历史','政治','数学']

class Config():
    data_dir = FLAGS.data_path
    vali_rate = FLAGS.vali_rate
    rnn_size = FLAGS.hidden_neural_size
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    seq_length = FLAGS.seq_length
    embed_dim = FLAGS.embedding_dim
    vocab_size = FLAGS.vocabulary_size
    layer_num = FLAGS.hidden_layer_num
    class_num = FLAGS.class_num
    keep_prob = FLAGS.keep_prob
    num_epoch = FLAGS.num_epoch
    check_point_every = FLAGS.check_point_every
    num_step = FLAGS.seq_length
    init_scale = FLAGS.init_scale
    save_dir = FLAGS.save
    init_from = FLAGS.init_from_model
    log_dir = FLAGS.log
    max_decay_epoch = FLAGS.max_decay_epoch


def train():
    #initial config
    config = Config()
    if config.init_from:
        ckpt = tf.train.get_checkpoint_state(config.init_from)
        assert ckpt,'No Checketpoint Found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        #with open(os.path.join(config.save_dir,'config.pkl'),'r')as rf:
        #    config = cPickle.load(rf)
    else:
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        with open(os.path.join(config.save_dir,'config.pkl'),'wb') as wf:
            cPickle.dump(config,wf)
        if not os.path.isdir(config.log_dir):
            os.makedirs(config.log_dir)

    dataloader = dataLoader(config.batch_size,config.vocab_size,config.data_dir,\
            config.seq_length,config.vali_rate)
    #config.vocab_size = dataloader.vocab_size
    gpu_option = tf.GPUOptions(allow_growth=True)
    sessconfig = tf.ConfigProto(gpu_options=gpu_option)
    with tf.Session(config=sessconfig) as sess:
        initializer = tf.random_uniform_initializer(-1*config.init_scale,1*config.init_scale)
        with tf.variable_scope('model',reuse=None,initializer=initializer):
            model = LstmModel(config)

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(config.log_dir,time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if config.init_from:
            print 'load model'
            saver.restore(sess,ckpt.model_checkpoint_path)
        #train
        num_batches = dataloader.train_num
        for e in range(config.num_epoch):
            lr_decay = config.lr_decay ** max(e-config.max_decay_epoch,0.0)
            model.assign_new_lr(sess,config.lr*lr_decay)
            for i,(x,y,mask) in enumerate(dataloader.get_batches('train')):
                start = time.time()
                feed = {model.input_data:x,model.targets:y,model.mask:mask,\
                        model.keep_prob:config.keep_prob}
                state = sess.run(model._initial_state)
                for j , (c,h) in enumerate(model._initial_state):
                    feed[c]=state[j].c
                    feed[h]=state[j].h
                loss,acc,summ,_=sess.run([model.cost,model.accuracy,summaries,model.train_op],feed)
                #print len(midoput)
                #print midoput[0].shape
                writer.add_summary(summ,e*config.batch_size+i)
                end = time.time()
                print("{}/{} (epoch {}), train_loss={:.5f},acc={:.5f},time/batch ={:.4f}"\
                        .format(e * num_batches + i,\
                        config.num_epoch*num_batches,e,loss, acc,end - start))
                if (config.num_epoch*num_batches+i)%config.check_point_every==0\
                        or (e==config.num_epoch-1 and i ==num_batches-1):
                    checkpoint_path = os.path.join(config.save_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step=e*num_batches+i)
            validation(dataloader,sess,model)


def validation(dataloader,sess,model):
        #validation
        total_loss=0.0
        total_acc=0.0
        #sen = [line.strip() for line in open('data/sen.txt','r')]
        index = 0
        wf = open('error.txt','wb')
        vocab={}
        for k,v in dataloader.vocab.items():
            vocab[v]=k
        for i,(x,y,mask) in enumerate(dataloader.get_batches('validation')):
            start = time.time()
            feed = {model.input_data:x,model.targets:y,model.mask:mask,\
                    model.keep_prob:1.0}
            loss,acc,pre = sess.run([model.cost,model.accuracy,model.pre],feed)
            total_loss += loss
            total_acc += acc
            for v in range(len(y)):
                if y[v]!=pre[v]:
                    sen = ''.join([vocab[w]+' ' for w in x[v] if w!=0])
                    wf.write(sen+'\t'+kemu[y[v]]+'\t'+kemu[pre[v]]+'\n')
                index += 1
            end = time.time()
            print ("val_batch{},vali_loss={:.7f},vali_acc={:.5f},time/batch={:.4f}"\
                    .format(i+1,loss,acc,end-start))
        print ('average loss is %.7f'%(total_loss/dataloader.vali_num))
        print ('average acc is %.7f'%(total_acc/dataloader.vali_num))
        sleep(1)

def main(_):
    train()

if __name__=='__main__':
    tf.app.run()

