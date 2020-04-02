import os
import pandas as pd

from tqdm import tqdm


def label2csv(input_file, output_file, labels_file):

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
            label[labels.index(sample_label)] = 1

        sample_items = sample_items[:3] + label
        data.append(sample_items)

    columns = ['file', 'age', 'sex'] + labels
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(output_file, index=False)
    return


def merge_train_testA(train_csv, train_dir, testA_csv, testA_dir,
                      output_csv):
    train_df = pd.read_csv(train_csv)
    testA_df = pd.read_csv(testA_csv)

    train_dirs = [train_dir] * len(train_df)
    testA_dirs = [testA_dir] * len(testA_df)

    train_df.insert(1, 'ecg_dir', train_dirs)
    testA_df.insert(1, 'ecg_dir', testA_dirs)

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


def remove_duplicates(train_testA_csv, output_csv):

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


def main(args):

    print('=' * 100)
    print('Preprocessing on train and testA')
    print('-' * 100)

    # -1- Convert labels of train and testA to csv
    print('-1- Convert labels of train and testA to csv')
    train_csv = os.path.join(args.output_dir, 'train.csv')
    label2csv(args.train_txt, train_csv, args.arrythmia_txt)

    testA_csv = os.path.join(args.output_dir, 'testA.csv')
    label2csv(args.testA_txt, testA_csv, args.arrythmia_txt)

    # -2- Merge labels of train and testA
    print('-2- Merge labels of train and testA')
    train_testA_csv = os.path.join(args.output_dir, 'train_test.csv')
    merge_train_testA(
        train_csv, args.train_dir, testA_csv, args.testA_dir, train_testA_csv
    )

    # -3- Remove duplicates in train and testA
    print('-3- Remove duplicates in train and testA')
    train_testA_noDup_csv = os.path.join(args.output_dir, 'train_testA_noDup.csv')
    remove_duplicates(train_testA_csv, train_testA_noDup_csv)

    print('=' * 100, '\n')
    return


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='HFECG Competition -Round 1- Preprocessing Pipeline'
    )

    parser.add_argument('--arrythmia-txt', '-a', type=str,
                        action='store', dest='arrythmia_txt',
                        help='File provides all labels')
    parser.add_argument('--train-txt', '-tt', type=str,
                        action='store', dest='train_txt',
                        help='Labels of trainig set')
    parser.add_argument('--testA-txt', '-at', type=str,
                        action='store', dest='testA_txt',
                        help='Labels of testA set')
    parser.add_argument('--train-dir', '-td', type=str,
                        action='store', dest='train_dir',
                        help='Directory of training set')
    parser.add_argument('--testA-dir', '-ad', type=str,
                        action='store', dest='testA_dir',
                        help='Directory of testA set')
    parser.add_argument('--output-dir', '-o', type=str,
                        action='store', dest='output_dir',
                        help='Directory to save preprocessed data')

    args = parser.parse_args()
    main(args)
