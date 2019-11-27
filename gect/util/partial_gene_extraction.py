#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:08:10 2019

@author: heavens
"""
import pandas as pd

store_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/all_data.h5"
partial_file = "/home/heavens/CMU/Semester1/Machine_Learning_10-701/Project/ml_10701_ps5_data/partial_gene.h5"
store = pd.HDFStore(store_file)
feature_matrix_dataframe = store['rpkm']
labels_series = store['labels']
accessions_series = store['accessions']
feature_matrix_dataframe.iloc[:,0:100].to_hdf(partial_file,key = 'rpkm')
labels_series.to_hdf(partial_file,key = 'labels')
accessions_series.to_hdf(partial_file,key = 'accessions')
store.close()
partial_store = pd.HDFStore(partial_file)
partial_feature = partial_store['rpkm']
print(partial_feature.shape)