#!/bin/sh

# Preprocessing
# -1- Convert labels of train set to csv
python ./prep/ecg_prep.py -i ../data -o ../user_data

# ------------------------------------------------------

# Training model
# 5 Cross-Validation Trining

# small model
python ./train/ecg_train.py \
    -l2 ../user_data/round2_train_weighted.csv \
    -l1 ../user_data/round1_merge_noDup_weighted.csv \
    -m ../user_data/models2_small \
    -b small -nt 6 -c 5 -g 0

# large model
# python ./train/ecg_train.py \
#     -l2 ../user_data/round2_train_weighted.csv \
#     -l1 ../user_data/round1_merge_noDup_weighted.csv \
#     -m ../user_data/models2_large \
#     -b large -nt 6 -c 5 -g 0

# ------------------------------------------------------

# Testing model
# Ensemble 5 predictions

python ./test/ecg_test.py \
    -s /tcdata/hf_round2_testB \
    -m ../user_data/models2 \
    -t /tcdata/hf_round2_subB.txt \
    -a ../data/hf_round2_arrythmia.txt \
    -o ../prediction_result \
    -b small -g 0
