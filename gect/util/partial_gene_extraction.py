#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:08:10 2019

@author: heavens
"""
import pandas as pd
import numpy as np

store_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/train_data.h5"
partial_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/train_part.h5"
rest_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/train_rest.h5"
store = pd.HDFStore(store_file)
feature_matrix_dataframe = store['rpkm']
labels_series = store['labels']
accessions_series = store['accessions']
partial_number = np.int(np.floor(feature_matrix_dataframe.shape[0]*0.9))
perm = np.random.permutation(feature_matrix_dataframe.shape[0])
part_idx = perm[:partial_number]
rest_idx = perm[partial_number:]
feature_matrix_dataframe.iloc[part_idx,:].to_hdf(partial_file,key = 'rpkm')
labels_series.iloc[part_idx].to_hdf(partial_file,key = 'labels')
accessions_series.iloc[part_idx].to_hdf(partial_file,key = 'accessions')

feature_matrix_dataframe.iloc[rest_idx,:].to_hdf(rest_file,key = 'rpkm')
labels_series.iloc[rest_idx].to_hdf(rest_file,key = 'labels')
accessions_series.iloc[rest_idx].to_hdf(rest_file,key = 'accessions')
store.close()
partial_store = pd.HDFStore(partial_file)
partial_feature = partial_store['rpkm']
print(partial_feature.shape)
