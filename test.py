import pandas as pd
import numpy as np
import tensorflow as tf
import ctrNet
from sklearn.model_selection import train_test_split
from src import misc_utils as utils
import os
train_df=pd.read_csv('data/train_small.txt',header=None,sep='\t')
train_df.columns=['label']+['f'+str(i) for i in range(39)]
train_df, dev_df,_,_ = train_test_split(train_df,train_df,test_size=0.1, random_state=2019)
dev_df, test_df,_,_ = train_test_split(dev_df,dev_df,test_size=0.5, random_state=2019)
features=['f'+str(i) for i in range(39)]


#FM
hparam=tf.contrib.training.HParams(
            model='fm', #['fm','ffm','nffm']
            k=16,
            hash_ids=int(1e5),
            batch_size=64,
            optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
            learning_rate=0.0002,
            num_display_steps=100,
            num_eval_steps=1000,
            steps=200,
            epoch=2,
            metric='auc', #['auc','logloss']
            init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
            init_value=0.1,
            feature_nums=len(features))
utils.print_hparams(hparam)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='7'
model=ctrNet.build_model(hparam)
print("Testing FM....")
model.train(train_data=(train_df[features],train_df['label']),\
            dev_data=(dev_df[features],dev_df['label']))
from sklearn import metrics
preds=model.infer(dev_data=(test_df[features],test_df['label']))
fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
auc=metrics.auc(fpr, tpr)
print(auc)

print("FM Done....")




#FFM
hparam=tf.contrib.training.HParams(
            model='ffm', #['fm','ffm','nffm']
            k=16,
            hash_ids=int(1e5),
            batch_size=64,
            optimizer="adam", #['adadelta','adagrad','sgd','adam','ftrl','gd','padagrad','pgd','rmsprop']
            learning_rate=0.0002,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=2,
            metric='auc', #['auc','logloss']
            init_method='uniform', #['tnormal','uniform','normal','xavier_normal','xavier_uniform','he_normal','he_uniform']
            init_value=0.1,
            feature_nums=len(features))
utils.print_hparams(hparam)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='7'
model=ctrNet.build_model(hparam)
print("Testing FFM....")
model.train(train_data=(train_df[features],train_df['label']),\
            dev_data=(dev_df[features],dev_df['label']))
from sklearn import metrics
preds=model.infer(dev_data=(test_df[features],test_df['label']))
fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
auc=metrics.auc(fpr, tpr)
print(auc)

print("FFM Done....")

#NFFM
hparam=tf.contrib.training.HParams(
            model='nffm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            cross_layer_sizes=[128,128,128],
            k=8,
            hash_ids=int(2e5),
            batch_size=64,
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=1,
            metric='auc',
            activation=['relu','relu','relu'],
            cross_activation='identity',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features))
utils.print_hparams(hparam)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='7'
model=ctrNet.build_model(hparam)

print("Testing NFFM....")
model.train(train_data=(train_df[features],train_df['label']),\
            dev_data=(dev_df[features],dev_df['label']))
from sklearn import metrics
preds=model.infer(dev_data=(test_df[features],test_df['label']))
fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
auc=metrics.auc(fpr, tpr)
print(auc)

print("NFFM Done....")


#Xdeepfm
hparam=tf.contrib.training.HParams(
            model='xdeepfm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[128,128],
            cross_layer_sizes=[128,128,128],
            k=8,
            hash_ids=int(2e5),
            batch_size=64,
            optimizer="adam",
            learning_rate=0.001,
            num_display_steps=100,
            num_eval_steps=1000,
            epoch=1,
            metric='auc',
            activation=['relu','relu','relu'],
            cross_activation='identity',
            init_method='uniform',
            init_value=0.1,
            feature_nums=len(features))
utils.print_hparams(hparam)
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='7'
model=ctrNet.build_model(hparam)
print("Testing XdeepFM....")
model.train(train_data=(train_df[features],train_df['label']),\
            dev_data=(dev_df[features],dev_df['label']))
from sklearn import metrics
preds=model.infer(dev_data=(test_df[features],test_df['label']))
fpr, tpr, thresholds = metrics.roc_curve(test_df['label']+1, preds, pos_label=2)
auc=metrics.auc(fpr, tpr)
print(auc)

print("XdeepFM Done....")

