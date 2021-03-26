#!/bin/bash
source activate pytorch-13
export PYTHONPATH=~/CMU/GECT/
python ~/CMU/GECT/gect/gect_train_embedding.py --train_data ~/CMU/ml_10701_ps5_data/all_data.h5 --log-dir ~/CMU/GECT/gect/ -m embedding_model_test -t 1e-1 --epoches 1000
