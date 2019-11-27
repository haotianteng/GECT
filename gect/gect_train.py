#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 03:23:25 2019

@author: Haotian Teng
"""

import torch
import numpy as np
import os 

class Trainer(object):
    def __init__(self,train_dataloader,net,keep_record = 5,eval_dataloader = None,device = None):
        """Trainer
        Args:
            train_dataloader: A torch.utils.data.dataloader.DataLoader instance.
            net: A Neural network model.
            keep_record: Type Int, defualt is 5, the latest n checkpoints to save for each training routine.
            eval_dataloader: A torch.utils.data.dataloader.DataLoader instance, if None, use training dataloader.
            device: The device to put the model on, can be cpu or cuda, if None then gpu will be used if available.
        """
        self.train_ds = train_dataloader
        self.device = self._get_device(device)
        if eval_dataloader is None:
            self.eval_ds = self.train_ds
        else:
            self.eval_ds = eval_dataloader
        self.net = net
        self.global_step = 0
        self.keep_record = keep_record
        self.save_list = []
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
        
    def save(self,save_folder=None):
        if save_folder is not None:
            self.save_folder = save_folder
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        current_ckpt = 'ckpt-'+str(self.global_step)
        model_file = os.path.join(self.save_folder,current_ckpt)
        self.save_list.append(current_ckpt)
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self.save_list) > self.keep_record:
            os.remove(os.path.join(self.save_folder,self.save_list[0]))
            self.save_list = self.save_list[1:]
        with open(ckpt_file,'w+') as f:
            f.write("latest checkpoint:" + current_ckpt + '\n')
            for path in self.save_list:
                f.write("checkpoint file:" + path + '\n')
        torch.save(self.net.state_dict(),model_file)
    
    def load(self,save_folder):
        self.save_folder = save_folder
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            self.global_step = int(latest_ckpt.split('-')[1])
        self.net.load_state_dict(torch.load(os.path.join(save_folder,latest_ckpt),map_location=self.device))
        
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.net.to(self.device)
        self.save_folder = save_folder
        train_record = []
        valid_record = []
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if i_batch%save_cycle==0:
                    calculate_error = True
                else:
                    calculate_error = False
                loss,error = self.train_step(batch,get_error = calculate_error)
                optimizer.zero_grad()
                loss.backward()
                if i_batch%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error = self.valid_step(valid_batch)
                    mean_error = np.mean(error)
                    mean_valid_error = np.mean(valid_error)
                    print("Epoch %d Batch %d, loss %f, error %f, valid_error %f"%(epoch_i, i_batch, loss,mean_error,mean_valid_error))
                    train_record.append(mean_error)
                    valid_record.append(mean_valid_error)
                optimizer.step()
                self.global_step +=1
        return train_record,valid_record
    def train_step(self,batch,get_error = False):
        """Training step
        Input Args:
            batch: A input batch of data.
            get_error: If the error valid step going to be calculated.
        The train step is reimplemented according to different requirement.
        """
        pass
    
    def valid_steo(self,batch):
        pass
    
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device = None):
        self.dataloader = dataloader
        if device is None:
            device = self.get_default_device()
        else:
            device = torch.device(device)
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield self._to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        return data.to(device, non_blocking=True)
    
    def get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    