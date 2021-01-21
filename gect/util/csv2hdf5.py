#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:38:31 2020
Transfer the FISH gene expression file(csv format) to hdf5 file.
@author: haotian teng
"""
import h5py

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
   parser.set_defaults(retrain=False)
   args = parser.parse_args()
