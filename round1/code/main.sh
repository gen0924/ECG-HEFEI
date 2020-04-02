#!/bin/sh

# Preprocessing
# -1- Convert labels of train and testA to csv
# -2- Merge labels of train and testA
# -3- Remove duplicates in train and testA

python ./prep/ecg_prep.py \
    -a ../data/hf_round1_arrythmia.txt \
    -tt ../data/hf_round1_label.txt \
    -at ../data/hefei_round1_ansA_20191008.txt \
    -td ../data/hf_round1_train/train \
    -ad ../data/hf_round1_testA/testA \
    -o ../user_data

# ------------------------------------------------------

# Training model
# 5 Cross-Validation Training

# large model
python ./train/ecg_train.py \
    -l ../user_data/train_testA_noDup.csv \
    -m ../user_data/models_large \
    -b large -nt 6 -c 5 -g 0

# small model
# python ./train/ecg_train.py \
#     -l ../user_data/train_testA_noDup.csv \
#     -m ../user_data/models_small \
#     -b small -nt 6 -c 5 -g 0

# ------------------------------------------------------

# Testing model
# Ensemble 5 predictions

# large model
python ./test/ecg_test.py \
    -s ../data/hf_round1_testB_noDup_rename/testB_noDup_rename \
    -m ../user_data/models_large \
    -t ../data/hf_round1_subB_noDup_rename.txt \
    -a ../data/hf_round1_arrythmia.txt \
    -o ../prediction_result \
    -b large -g 0

# small model
# python ./test/ecg_test.py \
#     -s ../data/hf_round1_testB_noDup_rename/testB_noDup_rename \
#     -m ../user_data/models_small \
#     -t ../data/hf_round1_subB_noDup_rename.txt \
#     -a ../data/hf_round1_arrythmia.txt \
#     -o ../prediction_result \
#     -b small -g 0
