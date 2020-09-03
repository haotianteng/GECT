#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:50:54 2019

@author: Haotian Teng
"""

import h5py
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np

class GeneEmbedding(nn.Module):
    """
    The GeneEmbedding model.
    
    """
    def __init__(self,
                 gene_n,
                 embedding_size,
                 dropout_prob):
        super(GeneEmbedding, self).__init__()
        self.net = []
        self.linear1 = nn.Linear(gene_n,embedding_size,bias = True)
        self.net.append(self.linear1)
        self.linear2 = nn.Linear(embedding_size,gene_n,bias = True)
        self.net.append(self.linear2)
#        self.celoss = nn.BCELoss()
        self.l2loss = nn.MSELoss(reduction = 'mean')
        self.dropout_prob = dropout_prob
    def forward(self, feature,training = True):
        """The formward step for the afterward segment input.
        Args:
            Feature: A batch of the gene expression level has the shape of [Batch_size,gene_number]
        Output:
             Out: Reconstruction matrix with the shape [Batch_size,Segment_length]
        """
        feature = nn.functional.dropout(feature, p=self.dropout_prob,training = training)
#        for layer in self.net:
#            feature = layer(feature)
        feature = self.linear1(feature)
        feature = torch.matmul(feature,self.linear1.weight)
        return feature
    
    def MAPE_loss(self,predict,target):
        """The mean absolute percentage error
        """
        target_mean = torch.mean(target)
        if target_mean == 0:
            target_mean += 1e-6
        loss = torch.mean(torch.abs(target - predict)) / target_mean
        return loss
    
    def loss(self, predict, target):
        """
        The reconstruction loss
        """
        loss = self.l2loss(predict,target)
        return loss
    
    def error(self, predict, target):
        l2loss = self.l2loss(predict,target)
        l2loss = l2loss.detach().cpu().numpy()
        l2loss = np.mean(l2loss)
        return l2loss

class CellClassifier(nn.Module):
    """
    The Multiple layer forward network for cell typing.
    """
    def __init__(self,
                 embedding_size,
                 hidden_ns,
                 cell_n):
        super(CellClassifier, self).__init__()
        self.net = {}
        self.embedding_size = embedding_size
        self.hidden_ns = hidden_ns
        current_n = embedding_size
        for layer_idx,hidden_n in enumerate(hidden_ns):
            setattr(self,'linear'+str(layer_idx),nn.Linear(current_n,hidden_n,bias = True))
            self.net['linear'+str(layer_idx)] = getattr(self,'linear'+str(layer_idx))
            setattr(self,'skip'+str(layer_idx),nn.Linear(current_n,hidden_n,bias = False))
            self.net['skip'+str(layer_idx)] = getattr(self,'skip'+str(layer_idx))
            current_n = hidden_n
        setattr(self,'out_linear',nn.Linear(current_n,cell_n,bias = True))
        self.net['out_linear'] = getattr(self,'out_linear')
        self.CEloss = nn.CrossEntropyLoss()
    def forward(self, feature,training = True):
        """The formward step for the afterward segment input.
        Args:
            Feature: A batch of the gene expression level has the shape of [Batch_size,gene_number]
        Output:
             Out: Reconstruction matrix with the shape [Batch_size,Segment_length]
        """
        tanh = nn.Tanh()
        for layer_idx in np.arange(len(self.hidden_ns)):
            branch1 = self.net['linear'+str(layer_idx)](feature)
            branch1 = tanh(branch1)
            branch2 = self.net['skip'+str(layer_idx)](feature)
            feature = branch1 + branch2
#            feature = branch1
        feature = self.net['out_linear'](feature)
        return feature
    
    def loss(self, predict, target):
        """
        The reconstruction cross-entropy loss
        """
        loss = self.CEloss(predict,target)
        loss = torch.mean(loss)
        return loss
        
    def error(self, predict, target):
        predict = predict.argmax(1)
        compare = (predict == target)
        compare = compare.cpu().numpy()
        error = 1 - compare
        error = np.mean(error)
        return error