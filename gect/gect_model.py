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
    The convolutional segmentation machine.
    
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
        for layer in self.net:
            feature = layer(feature)
        return feature
    
    def loss(self, predict, target):
        """
        The reconstruction cross-entropy loss
        """
        loss = self.l2loss(predict,target)
        loss = torch.mean(loss)
        return loss
    
    def error(self, predict, target):
        l2loss = self.l2loss(predict,target)
        l2loss = l2loss.detach().cpu().numpy()
        l2loss = np.mean(l2loss)
        return l2loss

