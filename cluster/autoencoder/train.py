#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
from dAutoencoder import AutoEncoder
import cPickle


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size',100,'the batch_size of the training procedure')
flags.DEFINE_integer('n_input',228,'the size of input')
flags.DEFINE_float('lr',0.001,'the learning rate')
flags.DEFINE_integer('n_hidden1',512,'hidden1 neural size')
flags.DEFINE_integer('n_hidden2',128,'hidden2 layer num')
flags.DEFINE_string('data_path','data','dataset path')
flags.DEFINE_float('keep_prob',0.96,'dropout rate')
flags.DEFINE_integer('num_epoch',10000,'num epoch')
flags.DEFINE_integer('check_point_every',10000,'checkpoint every num epoch ')
flags.DEFINE_string('init_from_model',None,'last model path')
flags.DEFINE_string('save','merge','model save path')
flags.DEFINE_string('log','log','log path')
flags.DEFINE_string('imgpath','imgfeature.npy','img path')
flags.DEFINE_string('txtpath','w2vfeature100.npy','img path')


class Config():
    data_dir = FLAGS.data_path
    n_hidden_1 = FLAGS.n_hidden1
    n_hidden_2 = FLAGS.n_hidden2
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    keep_prob = FLAGS.keep_prob
    num_epoch = FLAGS.num_epoch
    check_point_every = FLAGS.check_point_every
    init_from = FLAGS.init_from_model
    save_dir = FLAGS.save
    log_dir = FLAGS.log
    img_path = os.path.join(data_dir,FLAGS.imgpath)
    txt_path = os.path.join(data_dir,FLAGS.txtpath)
    n_input = FLAGS.n_input



def train():
    config = Config()

    if config.init_from:
        ckpt = tf.train.get_checkpoint_state(config.init_from)
        assert ckpt,'No Checketpoint Found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        with open(os.path.join(config.save_dir,'config.pkl'),'r')as rf:
            config = cPickle.load(rf)
    else:
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        with open(os.path.join(config.save_dir,'config.pkl'),'wb') as wf:
            cPickle.dump(config,wf)
        if not os.path.isdir(config.log_dir):
            os.makedirs(config.log_dir)

    model = AutoEncoder(config)
    model.build()
    gpu_option = tf.GPUOptions(allow_growth=True)
    sessconfig = tf.ConfigProto(gpu_options=gpu_option)
    with tf.Session(config=sessconfig) as sess:
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(\
                os.path.join(config.log_dir,time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if config.init_from is not None:
            saver.restore(sess,ckpt.model_checkpoint_path)

        imgfeature = np.load(config.img_path)
        txtfeature = np.load(config.txt_path)
        data = np.hstack((imgfeature,txtfeature))
        num_batches = int(len(data)/config.batch_size)
        data = data[:num_batches*config.batch_size]
        data = np.split(data,num_batches,0)
        for e in range(config.num_epoch):
            for i in range(num_batches):
                start = time.time()
                feed = {model.X:data[i]}
                loss,summ,_=sess.run([model.cost,summaries,model.optimizer],feed)
                writer.add_summary(summ,e*config.batch_size+i)
                end = time.time()
                print("{}/{} (epoch {}), train_loss={:.5f},time/batch ={:.4f}".format(\
                        e * num_batches + i,config.num_epoch*num_batches,e,loss,end - start))
                if (config.num_epoch*num_batches+i)%config.check_point_every==0\
                        or (e==config.num_epoch-1 and i ==num_batches-1):
                    checkpoint_path = os.path.join(config.save_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step=e*num_batches+i)
        imgfeature = np.zeros([num_batches*config.batch_size,config.n_hidden_2])
        index = 0
        for i in range(num_batches):
            feed = {model.X:data[i]}
            feature=sess.run(model.feature,feed)
            for j in feature:
                imgfeature[index] = j
                index += 1

        np.save(os.path.join(config.data_dir,'imgtxtfeature128.npy'),imgfeature)


def main(_):
    train()

if __name__=='__main__':
    tf.app.run()



