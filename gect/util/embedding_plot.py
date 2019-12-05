#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:03:17 2019

@author: heavens
"""

from sklearn.manifold import TSNE
import lightgbm as lgb
from gect import gect_input as gi
import os 
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import time

def load_embedding(model_folder):
    ckpt_file = os.path.join(model_folder,'checkpoint') 
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    state_dict = torch.load(os.path.join(model_folder,latest_ckpt))
    embedding_matrix = state_dict['linear1.weight'].detach().cpu().numpy()
    return embedding_matrix.transpose()

def error(predict,target,train_label_tags,test_label_tags):
    predict = train_label_tags[predict]
    target = test_label_tags[target]
    error = np.sum(predict!=target)/len(target)
    return error

sub_cell_n = 7
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
x = d_full.feature
x = np.matmul(x,embedding)
uniq,counts = np.unique(d_full.labels[:,0],return_counts =True)
sub_cell_list = uniq[np.argsort(counts)]
sub_cell_list = sub_cell_list[-sub_cell_n:]
sub_cell_choice = d_full.labels[:,0]==sub_cell_list[0]
for t in sub_cell_list[1:]:
    sub_cell_choice = np.logical_or(sub_cell_choice,d_full.labels[:,0]==t)
#sub_cell_feature = x[sub_cell_choice]
sub_cell_feature = d_full.feature[sub_cell_choice]
sub_cell_label = np.asarray(d_full.labels[:,0][sub_cell_choice],dtype = np.object)
for t in sub_cell_list:
    sub_cell_label[sub_cell_label==t] = d_full.label_tags[t]
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(sub_cell_feature)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df_subset = {}
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['y'] = sub_cell_label
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 7),
    data=df_subset,
    legend='full',
    alpha=1.0,
    s=5,
    linewidth=0,
)