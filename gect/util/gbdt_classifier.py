#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 05:33:25 2019

@author: heavens
"""

import lightgbm as lgb
from gect import gect_input as gi
import os 
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch

def error(predict,target,train_label_tags,test_label_tags):
    predict = train_label_tags[predict]
    target = test_label_tags[target]
    error = np.sum(predict!=target)/len(target)
    return error

sub_cell_n = 10
batch_size = 200
device = "cuda"
learning_rate = 4e-3
epoches = 100
global_step = 0
COUNT_CYCLE = 10
retrain = False
root_dir = '/home/heavens/CMU/GECT/'
data_dir = '/home/heavens/CMU/GECT/data'
all_dat = os.path.join(data_dir,"all_data.h5")
train_dat = os.path.join(data_dir,"train_data.h5")
eval_dat = os.path.join(data_dir,"test_data.h5")

embedding_model = os.path.join(root_dir,"gect/embedding_model/")
embedding = gi.load_embedding(embedding_model)
test_model = os.path.join(root_dir,"gect/gbdt2/")
d_full = gi.dataset(all_dat,transform=transforms.Compose([gi.ToTensor()]))
d1 = gi.dataset(train_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                        gi.Embedding(embedding),
                                                        gi.ToTensor()]))
d2 = gi.dataset(eval_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                       gi.Embedding(embedding),
                                                        gi.ToTensor()]))
x = d1.feature
y = d1.labels[:,0]
x_test = d2.feature
y_test = d2.labels[:,0]
y_test = gi.transfer_label_tags(y_test,d2.label_tags,d1.label_tags)
x = np.matmul(x,embedding)
x_test = np.matmul(x_test,embedding)
sub_cell_list = gi.cell_list(y,sub_cell_n)
x_sub,y_sub = gi.get_sub_data(x,y,sub_cell_list)
x_sub_test,y_sub_test = gi.get_sub_data(x_test,y_test,sub_cell_list)

print('Loading data...')
# load or create your dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_sub, y_sub)
lgb_eval = lgb.Dataset(x_sub_test, y_sub_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'max_depth':-1,
    'min_child_samples': 20,
    'num_class':sub_cell_n
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=20)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(x_sub_test, num_iteration=gbm.best_iteration)
y_pred = np.argmax(y_pred,axis = 1)
# eval
print('The error of prediction is:', error(y_pred,y_sub_test,d1.label_tags,d1.label_tags))