#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 05:50:16 2019

@author: heavens
"""
from gect.gect_train import Trainer
from gect.gect_model import GeneEmbedding
from gect import gect_input as gi
from torchvision import transforms
import numpy as np
import torch
import os 
from torch.utils import data
from matplotlib import pyplot as plt


class EmbeddingTrainer(Trainer):
    """EmbeddingTrainer
    Args:
        train_dataloader: A torch.utils.data.dataloader.DataLoader instance.
        net: A Neural network model.
        keep_record: Type Int, defualt is 5, the latest n checkpoints to save for each training routine.
        eval_dataloader: A torch.utils.data.dataloader.DataLoader instance, if None, use training dataloader.
        device: The device to put the model on, can be cpu or cuda, if None then gpu will be used if available.
        input_drop: The droupout probability for the input data, 0 use all data.
        output_drop: The dropout probability for the output data, 0 use all output to calculate loss.
    """
    def __init__(self,
                 train_dataloader,
                 net,
                 keep_record = 5,
                 eval_dataloader = None,
                 device = None,
                 input_drop = 0,
                 output_drop = 0):
        super(EmbeddingTrainer,self).__init__(train_dataloader=train_dataloader,
                                              net=net,
                                              keep_record = keep_record,
                                              eval_dataloader = eval_dataloader,
                                              device = device)
        self.input_drop = input_drop
        self.output_drop = output_drop
        
    def valid_step(self,batch):
        feature_batch = batch['feature']
        out = self.net.forward(feature_batch,training = False)
        return self.net.error(out,feature_batch)

    def train_step(self,batch,get_error = False):
        feature_batch = batch['feature']
        feature_n = feature_batch.shape[1]
#        output_mask = np.random.choice(2,feature_n,p = [self.output_drop, 1-self.output_drop])
#        output_mask = np.asarray(output_mask,dtype = gi.FEATURE_DTYPE)
#        output_mask = torch.from_numpy(output_mask).expand_as(feature_batch).to(self.device)
        out = self.net.forward(feature_batch,training = True)
#        loss = self.net.loss(out*output_mask,feature_batch*output_mask)
        loss = self.net.loss(out,feature_batch)
        error = None
        if get_error:
            error = self.net.error(out,feature_batch)
        return loss,np.asarray(error)

if __name__ == "__main__":
    root_dir = '/home/heavens/CMU/GECT/'
    data_dir = '/home/heavens/CMU/GECT/data'
    train_dat = os.path.join(data_dir,"all_data.h5")
    eval_dat = os.path.join(data_dir,"test_data.h5")
#    train_dat = os.path.join(root_dir,'data/partial_gene.h5')
#    eval_dat = os.path.join(root_dir,'data/partial_gene.h5')
    test_model = os.path.join(root_dir,"gect/embedding_model/")
    drop_prob = 0.9
    learning_rate = 1e-4
    epoches = 100
    global_step = 0
    COUNT_CYCLE = 10
    embedding_size = 200
    batch_size = 100
    model_dtype = np.float
    
    d1 = gi.dataset(train_dat,transform=transforms.Compose([gi.MeanNormalization(),
                                                            gi.ToTensor()]))
    d2 = gi.dataset(eval_dat,transform=transforms.Compose([gi.MeanNormalization(),
                                                           gi.ToTensor()]))
    assert(d1.feature.shape[1] == d2.feature.shape[1])
    device = "cuda"
    dataloader = gi.DeviceDataLoader(data.DataLoader(d1,batch_size=batch_size,shuffle=True,num_workers=5),device = device)
    eval_dataloader = gi.DeviceDataLoader(data.DataLoader(d2,batch_size=batch_size,shuffle=False,num_workers=5),device = device)
    net = GeneEmbedding(d1.feature.shape[1],embedding_size,dropout_prob = drop_prob)
    if gi.FEATURE_DTYPE == np.float32:
        net = net.float()
    elif gi.FEATURE_DTYPE == np.double:
        net = net.double()
    t = EmbeddingTrainer(train_dataloader = dataloader, 
                         net = net,
                         eval_dataloader = eval_dataloader,
                         device = device,
                         input_drop = drop_prob)
#    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#    try:
#        t.load(test_model)
#    except FileNotFoundError:
#        print("Model checkpoint file not found.")
#        pass
    train_record,valid_record = t.train(epoches,optimizer,COUNT_CYCLE,test_model)
    
    train_step = np.arange(0,epoches*len(d1),batch_size*COUNT_CYCLE)
    fig_h = plt.figure()
    axes = fig_h.add_axes([0.1,0.1,0.8,0.8])
    line1 = axes.plot(train_step,train_record,'r',label = 'train error')
    line2 = axes.plot(train_step,valid_record,'b',label = 'valid error')
    axes.legend()
    t.save(test_model)
