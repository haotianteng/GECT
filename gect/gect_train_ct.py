#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 05:46:29 2019

@author: heavens
"""
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
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)
model_folder = "/home/heavens/CMU/GECT/gect/embedding_model"

def load_embedding(model_folder):
    ckpt_file = os.path.join(model_folder,'checkpoint') 
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    state_dict = torch.load(os.path.join(model_folder,latest_ckpt))
    embedding_matrix = state_dict['linear1.weight'].detach().cpu().numpy()
    return embedding_matrix.transpose()

class CellTypingTrainer(Trainer):
    """CellTyperTrainer
    Args:
        train_dataloader: A torch.utils.data.dataloader.DataLoader instance.
        net: A Neural network model.
        keep_record: Type Int, defualt is 5, the latest n checkpoints to save for each training routine.
        eval_dataloader: A torch.utils.data.dataloader.DataLoader instance, if None, use training dataloader.
        device: The device to put the model on, can be cpu or cuda, if None then gpu will be used if available.
     """
    def __init__(self,
                 train_dataloader,
                 net,
                 keep_record = 5,
                 l1_regularizor = 1e-1,
                 l2_regularizor = 1e-1,
                 eval_dataloader = None,
                 device = None):
        super(CellTypingTrainer,self).__init__(train_dataloader=train_dataloader,
                                              net=net,
                                              keep_record = keep_record,
                                              eval_dataloader = eval_dataloader,
                                              device = device)
        self.l1_r = torch.tensor(l1_regularizor,device = self.device)
        self.l2_r = torch.tensor(l2_regularizor,device = self.device)
    def valid_step(self,batch):
        feature_batch = batch['feature']
        label_batch = batch['label']
        out = self.net.forward(feature_batch)
        return self.net.error(out,label_batch)

    def train_step(self,batch,get_error = False):
        feature_batch = batch['feature']
        label_batch = batch['label']
        out = self.net.forward(feature_batch)
        loss = self.net.loss(out,label_batch)
        l1loss = self.l1_r * self.l1loss()
        l2loss = self.l2_r * self.l2loss()
        loss = loss+ l1loss+l2loss
        if get_error:
            error = self.net.error(out,label_batch)
            return loss,np.asarray(error)
        else:
            return loss,None
        
    def l1loss(self):
        loss = torch.tensor(0.0,device = self.device)
        for p in self.net.parameters():
            loss += torch.norm(p,1)
        return loss
    
    def l2loss(self):
        loss = torch.tensor(0.0,device = self.device)
        for p in self.net.parameters():
            loss += torch.norm(p,2)
        return loss
    
batch_size = 100
device = "cuda"
net_structure = [100,40,20]
learning_rate = 2e-3
epoches = 100
global_step = 0
COUNT_CYCLE = 20
early_stop = 20
retrain = False
root_dir = '/home/heavens/CMU/GECT/'
data_dir = '/home/heavens/CMU/GECT/data'
all_dat = os.path.join(data_dir,"all_data.h5")
train_dat = os.path.join(data_dir,"train_part_cell.h5")
eval_dat = os.path.join(data_dir,"test_part_cell.h5")

#train_dat = os.path.join(data_dir,"train_part.h5")
#eval_dat = os.path.join(data_dir,"train_rest.h5")

embedding_model = os.path.join(root_dir,"gect/embedding_model/")
embedding = load_embedding(embedding_model)
test_model = os.path.join(root_dir,"gect/cell_classifier_10cell_2/")
d_full = gi.dataset(all_dat,transform=transforms.Compose([gi.ToTensor()]))
d1 = gi.dataset(train_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                        gi.Embedding(embedding),
                                                        gi.ToTensor()]))
d2 = gi.dataset(eval_dat,transform=transforms.Compose([gi.ToTags(d_full.label_tags),
                                                       gi.Embedding(embedding),
                                                        gi.ToTensor()]))
assert(d1.feature.shape[1] == d2.feature.shape[1])
cell_n = len(d_full.label_tags)

dataloader = gi.DeviceDataLoader(data.DataLoader(d1,batch_size=batch_size,shuffle=True,num_workers=5),device = device)
eval_dataloader = gi.DeviceDataLoader(data.DataLoader(d2,batch_size=len(d2.feature),shuffle=False,num_workers=1),device = device)
net = gm.CellClassifier(embedding_size = embedding.shape[1],
                   hidden_ns = net_structure,
                   cell_n = cell_n)
trainer = CellTypingTrainer(train_dataloader = dataloader,
                           net = net,
                           keep_record = early_stop,
                           eval_dataloader = eval_dataloader,
                           device = device)

#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if retrain:
    trainer.load(test_model)
else:
    print("Initailize model.")


train_record,valid_record = trainer.train(epoches,optimizer,COUNT_CYCLE,test_model,early_stop = early_stop)

train_step = np.arange(0,batch_size*COUNT_CYCLE*len(train_record),batch_size*COUNT_CYCLE)
fig_h = plt.figure()
axes = fig_h.add_axes([0.1,0.1,0.8,0.8])
line1 = axes.plot(train_step,train_record,'r',label = 'train error')
line2 = axes.plot(train_step,valid_record,'b',label = 'valid error')
plt.axvline(train_step[np.argmin(valid_record)],color = 'green')
plt.text(train_step[np.argmin(valid_record)]+100,min(valid_record)+0.2,'Best model',fontsize = 20)
plt.title("Trian-valid error of neural network on %d cell types"%(len(d1.label_tags)),fontsize = 20)
axes.legend()
trainer.save(test_model)


### Logistic Regression test:
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0,
#                         solver =  'lbfgs', 
#                         multi_class='multinomial',
#                         max_iter = 1000)
#embeded_X = np.matmul(d1.feature,embedding)
#clf.fit(embeded_X,d1.labels[:,0])
#embeded_X2 = np.matmul(d2.feature,embedding)
#predict = np.argmax(clf.predict_proba(embeded_X2),axis = 1)
#cell_predict = d1.label_tags[predict]
#cell_target = d2.label_tags[d2.labels]

### Paint first 2 principle components
#from sklearn.decomposition import PCA
for i_batch, sample_batched in enumerate(dataloader):
    feature = sample_batched['feature']
    label = sample_batched['label']
    break
#feature = feature.detach().cpu().numpy()
#norm_f = feature - np.mean(feature,axis=0,keepdims = True)
#pca = PCA(n_components = 3)
#transformed_f = pca.fit_transform(norm_f)
#label_int = np.asarray([np.where(x==1)[0][0] for x in label.detach().cpu().numpy()])
#cell_type_sample = 5
#color_palette = sns.color_palette("deep", 5)
#transformed_f = transformed_f[label_int<5]
#label_int = label_int[label_int<5]
#for sample_idx,sample in enumerate(transformed_f):
#    plt.plot(sample[1],sample[2],'.',color = color_palette[label_int[sample_idx]])