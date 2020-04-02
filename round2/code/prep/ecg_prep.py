import os
import numpy as np
import pandas as pd

from tqdm import *


def label2csv(input_file, output_file, labels_file, train_dir):

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [a.strip() for a in labels]
    n_labels = len(labels)

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = f.readlines()

    data = []
    for sample in samples:
        sample_items = sample.strip().split('\t')
        sample_labels = sample_items[3:]
        label = [0] * n_labels

        for sample_label in sample_labels:
            if sample_label in labels:
                label[labels.index(sample_label)] = 1

        if sum(label) > 0:
            sample_prefix = sample_items[:3]
            sample_prefix.insert(1, train_dir)
            sample_items = sample_prefix + label
            data.append(sample_items)

    columns = ['file', 'ecg_dir', 'age', 'sex'] + labels
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(output_file, index=False)
    return


def merge_round1_train_testA(train_csv, testA_csv, output_csv):
    train_df = pd.read_csv(train_csv)
    testA_df = pd.read_csv(testA_csv)

    columns = train_df.columns.tolist()
    train_list = train_df.values.tolist()
    testA_list = testA_df.values.tolist()
    train_files = [tl[0] for tl in train_list]

    for testA_item in testA_list:
        if testA_item[0] not in train_files:
            train_list.append(testA_item)

    train_testA_df = pd.DataFrame(data=train_list, columns=columns)
    train_testA_df.to_csv(output_csv, index=False)
    return


def remove_round1_duplicates(train_testA_csv, output_csv):

    def load_ecg(ecg_path):
        with open(ecg_path, 'r') as f:
            ecg = f.readlines()
        return ecg

    train_testA_df = pd.read_csv(train_testA_csv)
    columns = train_testA_df.columns.tolist()
    labels = train_testA_df.values.tolist()
    IDs = [label[0] for label in labels]

    print('Loading ECG ...')
    ecgs = []
    for label in tqdm(labels, ncols=75):
        ecgs.append(load_ecg(os.path.join(label[1], label[0])))

    print('Removing duplicates ...')
    duplicates, count = [], 0
    for i in tqdm(range(len(IDs)), ncols=75):
        if IDs[i] in duplicates:
            continue

        ecg1 = ecgs[i]
        for j in range(i + 1, len(IDs)):
            ecg2 = ecgs[j]
            if ecg1 == ecg2:
                if IDs[j] not in duplicates:
                    count += 1
                    duplicates.append(IDs[j])

    rm_dup_labels = []
    for label in labels:
        if label[0] not in duplicates:
            rm_dup_labels.append(label)

    rm_dup_df = pd.DataFrame(data=rm_dup_labels, columns=columns)
    rm_dup_df.to_csv(output_csv, index=False)
    return


def generate_sample_weights(round1_csv, round2_csv, round1_output_csv, round2_output_csv):

    round1 = pd.read_csv(round1_csv)
    round2 = pd.read_csv(round2_csv)
    round1_labels = round1.iloc[:, 4:]
    round2_labels = round2.iloc[:, 4:]

    round1_freq = round1_labels.sum() / len(round1_labels)
    round2_freq = round2_labels.sum() / len(round2_labels)
    round2_vs_round1 = round2_freq / round1_freq
    round1_labels_weights = round1_labels * round2_vs_round1

    round1_sample_weights = []
    for i, row in round1_labels_weights.iterrows():
        row_list = row.tolist()
        row_list = [item for item in row_list if item != 0.0]
        sample_weight = np.prod(row_list)
        round1_sample_weights.append(sample_weight)

    round1_sample_weights = np.array(round1_sample_weights)
    round1_sample_weights[round1_sample_weights > 1] = 1
    round1_sample_weights[round1_sample_weights < 0.3] = 0
    round1.insert(1, 'sample_weight', round1_sample_weights)
    drop_idxs = round1.loc[round1['sample_weight'] == 0.0].index.tolist()
    round1.drop(index=drop_idxs, inplace=True)

    round2_sample_weights = [1] * len(round2)
    round2.insert(1, 'sample_weight', round2_sample_weights)

    round1.to_csv(round1_output_csv, index=False)
    round2.to_csv(round2_output_csv, index=False)
    return


def main(args):

    print('=' * 100)
    print('Preprocessing on train set')
    print('-' * 100)

    round1_testA_dir = os.path.join(args.input_dir, 'hf_round1_testA')
    round1_train_dir = os.path.join(args.input_dir, 'hf_round1_train')
    round2_train_dir = os.path.join(args.input_dir, 'hf_round2_train')

    round1_testA_txt = os.path.join(args.input_dir, 'hefei_round1_ansA_20191008.txt')
    round1_train_txt = os.path.join(args.input_dir, 'hf_round1_label.txt')
    round2_train_txt = os.path.join(args.input_dir, 'hf_round2_train.txt')

    # round1_arrythmia_txt = os.path.join(args.input_dir, 'hf_round1_arrythmia.txt')
    round2_arrythmia_txt = os.path.join(args.input_dir, 'hf_round2_arrythmia.txt')

    # -1- Convert labels of dataset to csv
    print('-1- Convert labels of dataset to csv')
    round1_testA_csv = os.path.join(args.output_dir, 'round1_testA.csv')
    label2csv(round1_testA_txt, round1_testA_csv, round2_arrythmia_txt, round1_testA_dir)
    round1_train_csv = os.path.join(args.output_dir, 'round1_train.csv')
    label2csv(round1_train_txt, round1_train_csv, round2_arrythmia_txt, round1_train_dir)
    round2_train_csv = os.path.join(args.output_dir, 'round2_train.csv')
    label2csv(round2_train_txt, round2_train_csv, round2_arrythmia_txt, round2_train_dir)

    # -2- Merge train and testA of Round1
    print('-2- Merge train and testA of Round1')
    round1_merge_csv = os.path.join(args.output_dir, 'round1_merge.csv')
    merge_round1_train_testA(round1_train_csv, round1_testA_csv, round1_merge_csv)

    # -3- Remove duplicates in train and testA
    print('-3- Remove duplicates in train and testA of Round1')
    round1_merge_noDup_csv = os.path.join(args.output_dir, 'round1_merge_noDup.csv')
    remove_round1_duplicates(round1_merge_csv, round1_merge_noDup_csv)

    # -4- Generate sample weights for sampling
    print('-4- Generate sample weights for sampling')
    round1_sample_weights_csv = os.path.join(args.output_dir, 'round1_merge_noDup_weighted.csv')
    round2_sample_weights_csv = os.path.join(args.output_dir, 'round2_train_weighted.csv')
    generate_sample_weights(round1_merge_noDup_csv, round2_train_csv,
                            round1_sample_weights_csv, round2_sample_weights_csv)

    print('=' * 100, '\n')
    return


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='HFECG Competition -Round 2- Preprocessing Pipeline'
    )

    parser.add_argument('--input-dir', '-i', type=str,
                        action='store', dest='input_dir',
                        help='Directory of input data')
    parser.add_argument('--output-dir', '-o', type=str,
                        action='store', dest='output_dir',
                        help='Directory to save preprocessed data')

    args = parser.parse_args()
    main(args)
