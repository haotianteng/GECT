#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 05:10:10 2019

@author: heavens
"""

from sklearn.svm import LinearSVC as SVC
from gect import gect_model as gm
from gect.gect_train import Trainer
from gect import gect_input as gi
import torch.nn as nn
import os 
import torch
from torchvision import transforms
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def load_embedding(model_folder):
    ckpt_file = os.path.join(model_folder,'checkpoint') 
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    state_dict = torch.load(os.path.join(model_folder,latest_ckpt))
    embedding_matrix = state_dict['linear1.weight'].detach().cpu().numpy()
    return embedding_matrix.transpose()

batch_size = 200
device = "cuda"
net_structure = [50]
learning_rate = 4e-3
epoches = 100
global_step = 0
COUNT_CYCLE = 10
retrain = False
root_dir = '/home/heavens/CMU/GECT/'
data_dir = '/home/heavens/CMU/GECT/data'
all_dat = os.path.join(data_dir,"all_data.h5")
train_dat = os.path.join(data_dir,"train_part.h5")
eval_dat = os.path.join(data_dir,"test_data.h5")

embedding_model = os.path.join(root_dir,"gect/embedding_model/")
embedding = load_embedding(embedding_model)
test_model = os.path.join(root_dir,"gect/cell_classifier2/")
d_full = gi.dataset(all_dat,transform=transforms.Compose([gi.ToTensor()]))
d1 = gi.dataset(train_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                        gi.Embedding(embedding),
                                                        gi.ToTensor()]))
d2 = gi.dataset(eval_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                       gi.Embedding(embedding),
                                                        gi.ToTensor()]))

clf = SVC(random_state = 0, tol =1e-5)
x = d1.feature
print("Begin SVM training.")
x = np.matmul(x,embedding)
clf.fit(x,d1.labels[:,0])
x_test = d2.feature
x_test = np.matmul(x_test,embedding)
print("Begin predicting.")
predict = clf.predict(x_test)
predict = d1.label_tags[predict]
target = d2.label_tags[d2.labels[:,0]]
error = 1 - np.sum(target==predict)/len(target)
print("Error is %f"%(error))
