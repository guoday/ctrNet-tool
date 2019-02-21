import tensorflow as tf
from src import misc_utils as utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from models.base_model import BaseModel
import numpy as np
import time 
import os
class Model(BaseModel):
    def __init__(self,hparams):
        self.hparams=hparams
        if hparams.metric in ['logloss']:
            self.best_score=100000
        else:
            self.best_score=0
        self.build_graph(hparams)   
        self.optimizer(hparams)
        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))   
  
    def set_Session(self,sess):
        self.sess=sess
        
    def build_graph(self, hparams):
        initializer = self._get_initializer(hparams)
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        self.features=tf.placeholder(shape=(None,hparams.feature_nums), dtype=tf.int32)
        self.emb_v1=tf.get_variable(shape=[hparams.hash_ids,1],
                                    initializer=initializer,name='emb_v1')
        self.emb_v2=tf.get_variable(shape=[hparams.hash_ids,hparams.feature_nums,hparams.k],
                                    initializer=initializer,name='emb_v2')
        
        #lr
        emb_inp_v1=tf.gather(self.emb_v1, self.features)
        w1=tf.reduce_sum(emb_inp_v1,[-1,-2])
        
        
        emb_inp_v2=tf.gather(self.emb_v2, self.features)
        emb_inp_v2=tf.reduce_sum(emb_inp_v2*tf.transpose(emb_inp_v2,[0,2,1,3]),-1)
        temp=[]
        for i in range(hparams.feature_nums):
            if i!=0:
                temp.append(emb_inp_v2[:,i,:i])
        w2=tf.reduce_sum(tf.concat(temp,-1),-1)
        
        logit=w1+w2
        self.prob=tf.sigmoid(logit)
        logit_1=tf.log(self.prob+1e-20)
        logit_0=tf.log(1-self.prob+1e-20)
        self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        self.cost=-(self.label*logit_1+(1-self.label)*logit_0)
        self.saver= tf.train.Saver()
            
    def optimizer(self,hparams):
        opt=self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params)) 

    def train(self,train_data,dev_data=None):
        hparams=self.hparams
        sess=self.sess
        assert len(train_data[0])==len(train_data[1]), "Size of features data must be equal to label"
        for epoch in range(hparams.epoch):
            info={}
            info['loss']=[]
            info['norm']=[]
            start_time = time.time()
            for idx in range(len(train_data[0])//hparams.batch_size+3):
                try:
                    if hparams.steps<=idx:
                        T=(time.time()-start_time)
                        self.eval(T,dev_data,hparams,sess)
                        break               
                except:
                    pass                
                if idx*hparams.batch_size>=len(train_data[0]):
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
                    break
                    
                batch=train_data[0][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[0]))]
                batch=utils.hash_batch(batch,hparams)
                label=train_data[1][idx*hparams.batch_size:\
                                    min((idx+1)*hparams.batch_size,len(train_data[1]))]
                loss,_,norm=sess.run([self.loss,self.update,self.grad_norm],\
                                     feed_dict={self.features:batch,self.label:label})
                info['loss'].append(loss)
                info['norm'].append(norm)
                if (idx+1)%hparams.num_display_steps==0:
                    info['learning_rate']=hparams.learning_rate
                    info["train_ppl"]= np.mean(info['loss'])
                    info["avg_grad_norm"]=np.mean(info['norm'])
                    utils.print_step_info("  ", epoch,idx+1, info)
                    del info
                    info={}
                    info['loss']=[]
                    info['norm']=[]
                if (idx+1)%hparams.num_eval_steps==0 and dev_data:
                    T=(time.time()-start_time)
                    self.eval(T,dev_data,hparams,sess)
        self.saver.restore(sess,'model_tmp/model')
        T=(time.time()-start_time)
        self.eval(T,dev_data,hparams,sess)
        os.system("rm -r model_tmp")
        
      
    def infer(self,dev_data):
        hparams=self.hparams
        sess=self.sess
        assert len(dev_data[0])==len(dev_data[1]), "Size of features data must be equal to label"       
        preds=[]
        total_loss=[]
        for idx in range(len(dev_data[0])//hparams.batch_size+1):
            batch=dev_data[0][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[0]))]
            if len(batch)==0:
                break
            batch=utils.hash_batch(batch,hparams)
            label=dev_data[1][idx*hparams.batch_size:\
                              min((idx+1)*hparams.batch_size,len(dev_data[1]))]
            pred=sess.run(self.prob,\
                          feed_dict={self.features:batch,self.label:label})  
            preds.append(pred)   
        preds=np.concatenate(preds)
        return preds
            
