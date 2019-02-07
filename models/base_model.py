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
        self.iterator = iterator
        self.layer_params = []
        self.embed_params = []
        self.cross_params = []
        self.layer_keeps = None
        self.keep_prob_train = None
        self.keep_prob_test = None
        self.initializer = self._get_initializer(hparams)
        self.logit = self._build_graph(hparams)
        self.pred = self._get_pred(self.logit, hparams)
        self.data_loss = self._compute_data_loss(hparams)
        self.regular_loss = self._compute_regular_loss(hparams)
        self.loss = tf.add(self.data_loss, self.regular_loss)
        self.saver = tf.train.Saver(max_to_keep=hparams.epochs)
        self.update = self._build_train_opt(hparams)
        self.init_op = tf.global_variables_initializer()
        self.merged = self._add_summaries()

    def _get_pred(self, logit, hparams):
        if hparams.method == 'regression':
            pred = tf.identity(logit)
        elif hparams.method == 'classification':
            pred = tf.sigmoid(logit)
        else:
            raise ValueError("method must be regression or classification, but now is {0}".format(hparams.method))
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    @abc.abstractmethod
    def _build_graph(self, hparams):
        """Subclass must implement this."""
        pass

    def _l2_loss(self, hparams):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(l2_loss, tf.multiply(hparams.l2, tf.nn.l2_loss(param)))

        return l2_loss

    def _l1_loss(self, hparams):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(l1_loss, tf.multiply(hparams.embed_l1, tf.norm(param, ord=1)))
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(l1_loss, tf.multiply(hparams.layer_l1, tf.norm(param, ord=1)))
        return l1_loss

    def _cross_l_loss(self, hparams):
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(hparams.cross_l1, tf.norm(param, ord=1)))
            cross_l_loss = tf.add(cross_l_loss, tf.multiply(hparams.cross_l2, tf.norm(param, ord=1)))
        return cross_l_loss 

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

    def _compute_data_loss(self, hparams):
        if hparams.loss == 'cross_entropy_loss':
            data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                logits=tf.reshape(self.logit, [-1]), \
                labels=tf.reshape(self.iterator.labels, [-1])))
        elif hparams.loss == 'square_loss':
            data_loss = tf.sqrt(tf.reduce_mean(
                tf.squared_difference(tf.reshape(self.pred, [-1]), tf.reshape(self.iterator.labels, [-1]))))
        elif hparams.loss == 'log_loss':
            data_loss = tf.reduce_mean(tf.losses.log_loss(predictions=tf.reshape(self.pred, [-1]),
                                                          labels=tf.reshape(self.iterator.labels, [-1])))
        else:
            raise ValueError("this loss not defined {0}".format(hparams.loss))
        return data_loss

    def _compute_regular_loss(self, hparams):
        regular_loss = self._l2_loss(hparams) + self._l1_loss(hparams) + self._cross_l_loss(hparams)
        regular_loss = tf.reduce_sum(regular_loss)
        return regular_loss

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
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
