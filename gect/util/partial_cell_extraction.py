#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:48:02 2019

@author: heavens
"""
import pandas as pd
import numpy as np

def cell_list(y, cell_n):
    """Chooce cell_n cell types with most instances.
    """
    uniq,counts = np.unique(y,return_counts =True)
    sub_cell_list = uniq[np.argsort(counts)]
    sub_cell_list = sub_cell_list[-cell_n:]
    return sub_cell_list

def get_sub_data(x,y,sub_cell_list):
    sub_cell_choice = y==sub_cell_list[0]
    for t in sub_cell_list[1:]:
        sub_cell_choice = np.logical_or(sub_cell_choice,y==t)
    sub_x = x[sub_cell_choice]
    sub_y = y[sub_cell_choice]
    return sub_x,sub_y

def transfer_label_tags(y,label_tags_from,label_tags_to):
    labels = np.unique(y)
    new_y = np.copy(y)
    for label in labels:
        new_y[y==label] = np.where(label_tags_to == label_tags_from[label])[0][0]
    return new_y

sub_cell_n = 10

store_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/all_data.h5"
partial_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/test_part_cell.h5"
store = pd.HDFStore(store_file,'r')
feature_matrix_dataframe = store['rpkm']
labels_series = store['labels']
x = feature_matrix_dataframe
y = labels_series
sub_cell_list = cell_list(y,sub_cell_n)
print(sub_cell_list)
x_sub,y_sub = get_sub_data(x,y,sub_cell_list)
x_sub.to_hdf(partial_file,key = 'rpkm')
y_sub.to_hdf(partial_file,key = 'labels')
store.close()

store_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/train_data.h5"
partial_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/train_part_cell.h5"
store = pd.HDFStore(store_file,'r')
feature_matrix_dataframe = store['rpkm']
labels_series = store['labels']
x = feature_matrix_dataframe
y = labels_series
x_sub,y_sub = get_sub_data(x,y,sub_cell_list)
x_sub.to_hdf(partial_file,key = 'rpkm')
y_sub.to_hdf(partial_file,key = 'labels')
store.close()

