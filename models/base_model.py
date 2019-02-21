"""define base class model"""
import abc
import math
import tensorflow as tf
from sklearn import metrics
import os
from src import misc_utils as utils
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
__all__ = ["BaseModel"]


class BaseModel(object):
    def __init__(self, hparams,  scope=None):
        tf.set_random_seed(1234)
        
    @abc.abstractmethod
    def _build_graph(self, hparams):
        """Subclass must implement this."""
        pass


    def _get_initializer(self, hparams):
        if hparams.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'uniform':
            return tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
        elif hparams.init_method == 'normal':
            return tf.random_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif hparams.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif hparams.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=False)
        elif hparams.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=hparams.init_value)


    def _build_train_opt(self, hparams):
        def train_opt(hparams):
            if hparams.optimizer == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adagrad':
                train_step = tf.train.AdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'sgd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adam':
                train_step = tf.train.AdamOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'ftrl':
                train_step = tf.train.FtrlOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'gd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'padagrad':
                train_step = tf.train.ProximalAdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'pgd':
                train_step = tf.train.ProximalGradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer( \
                    hparams.learning_rate)
            else:
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            return train_step

        train_step = train_opt(hparams)
        return train_step
    
        
        
    def _active_layer(self, logit, scope, activation, layer_idx):
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, layer_idx):
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        return logit

    def train(self, sess):
        return sess.run([self.update, self.loss, self.data_loss, self.merged], \
                        feed_dict={self.layer_keeps: self.keep_prob_train})

    def eval(self,T,dev_data,hparams,sess):
        preds=self.infer(dev_data)
        if hparams.metric=='logloss':
            log_loss=metrics.log_loss(dev_data[1],preds)
            if self.best_score>log_loss:
                self.best_score=log_loss
                try:
                    os.makedirs('model_tmp/')
                except:
                    pass
                self.saver.save(sess,'model_tmp/model')
            utils.print_out("# Epcho-time %.2fs Eval logloss %.6f. Best logloss %.6f." \
                            %(T,log_loss,self.best_score))
        elif hparams.metric=='auc':
            fpr, tpr, thresholds = metrics.roc_curve(dev_data[1]+1, preds, pos_label=2)
            auc=metrics.auc(fpr, tpr)
            if self.best_score<auc:
                self.best_score=auc
                try:
                    os.makedirs('model_tmp/')
                except:
                    pass
                self.saver.save(sess,'model_tmp/model')                           
            utils.print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." \
                            %(T,auc,self.best_score))  

    def infer(self, sess):
        return sess.run([self.pred], \
                        feed_dict={self.layer_keeps: self.keep_prob_test})
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    

