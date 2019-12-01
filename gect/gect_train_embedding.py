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
import argparse
COUNT_CYCLE = 10
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
        out = self.net.forward(feature_batch,training = True)
        loss = self.net.loss(out,feature_batch)
        error = None
        if get_error:
            error = self.net.error(out,feature_batch)
        return loss,np.asarray(error)
    
def train_wrapper(args):
    train_dat = args.train_data
    if args.eval_data is None:
        eval_dat = train_dat
    else:
        eval_dat = args.eval_data
    test_model = os.path.join(args.log_dir,args.model_name)
    batch_size = args.batch_size
    d1 = gi.dataset(train_dat,transform=transforms.Compose([gi.MeanNormalization(),
                                                            gi.ToTensor()]))
    d2 = gi.dataset(eval_dat,transform=transforms.Compose([gi.MeanNormalization(),
                                                           gi.ToTensor()]))
    assert(d1.feature.shape[1] == d2.feature.shape[1])
    device = args.device
    dataloader = gi.DeviceDataLoader(data.DataLoader(d1,batch_size=batch_size,shuffle=True,num_workers=5),device = device)
    eval_dataloader = gi.DeviceDataLoader(data.DataLoader(d2,batch_size=batch_size,shuffle=False,num_workers=5),device = device)
    device = dataloader.device
    net = GeneEmbedding(d1.feature.shape[1],args.embedding_size,dropout_prob = args.drop_out)
    if gi.FEATURE_DTYPE == np.float32:
        net = net.float()
    elif gi.FEATURE_DTYPE == np.double:
        net = net.double()
    t = EmbeddingTrainer(train_dataloader = dataloader, 
                         net = net,
                         eval_dataloader = eval_dataloader,
                         device = device,
                         input_drop = args.drop_out)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.step_rate)
    if args.retrain:
        print("Load model from %s"%(test_model))
        t.load(test_model)
    else:
        print("Initialize model.")
    train_record,valid_record = t.train(args.epoches,optimizer,COUNT_CYCLE,test_model)
    train_step = np.arange(0,batch_size*COUNT_CYCLE*len(train_record),batch_size*COUNT_CYCLE)
    fig_h = plt.figure()
    axes = fig_h.add_axes([0.1,0.1,0.8,0.8])
    axes.plot(train_step,train_record,'r',label = 'train error')
    axes.plot(train_step,valid_record,'b',label = 'valid error')
    axes.legend()
    plt.savefig(os.path.join(test_model,'train_record.png'))
    t.save(test_model)
    
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument(
           '-i',
           '--train_data',
           help='Training file path',
           required=True)
   parser.add_argument(
           '-e',
           '--eval_data',
           default = None,
           help='Test file path')
   
   parser.add_argument(
           '-o',
           '--log-dir',
           help='Log dir location',
           required=True)
   
   parser.add_argument(
           '-m',
           '--model-name',
           help='Model name',
           required=True)
   
   parser.add_argument(
           '--embedding-size',
           help='Size of the embedding',
           default=200,
           type=int)
   parser.add_argument(
           '-b',
           '--batch-size',
           help='Training batch size',
           default=100,
           type=int)
   parser.add_argument(
           '-t',
           '--step-rate',
           help='Step rate',
           default=1e-4,
           type=float)
   parser.add_argument(
           '-d',
           '--drop-out',
           help='Dropout Probability',
           default=0.9,
           type=float)
   
   parser.add_argument(
           '--epoches',
           help='Max training epoches.',
           default=100,
           type=int)

   parser.add_argument(
            '--retrain', 
            dest='retrain', 
            action='store_true',
            help='Set retrain to true')
   parser.add_argument(
           '--device',
           help = "Device used to train, can be cpu or cuda.",
           default = None)
   
   parser.set_defaults(retrain=False)
   args = parser.parse_args()
   train_wrapper(args)
