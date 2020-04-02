import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


# Mean and Std of train set
ECG_MEAN = np.array(
    [0.618, 0.974, 0.080, 1.172, 1.415, 1.419,
     1.187, 0.954, 0.356, -0.796, 0.131, 0.665]
).reshape((-1, 1))
ECG_STD = np.array(
    [24.862, 33.086, 39.441, 62.491, 59.789, 64.328,
     58.257, 50.321, 25.534, 26.332, 19.010, 26.810]
).reshape((-1, 1))


class ECGDataset(Dataset):

    def __init__(self, dataset, is_train):
        super(ECGDataset, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        return

    def __len__(self):
        return len(self.dataset)

    def __augment(self, ecg):
        ecg_tmp = np.copy(ecg)
        channels, length = ecg.shape

        if np.random.randn() > 0.8:
            scale = np.random.normal(loc=1.0, scale=0.1, size=(channels, 1))
            scale = np.matmul(scale, np.ones((1, length)))
            ecg_tmp = ecg_tmp * scale

        if np.random.randn() > 0.8:
            for c in range(channels):
                offset = np.random.choice(range(-5, 5))
                ecg_tmp[c, :] += offset
        return ecg_tmp

    def __load_ecg(self, ecg_path):
        with open(ecg_path, 'r') as f:
            content = f.readlines()

        content = [list(map(int, c.strip().split())) for c in content[1:]]
        ecg = np.array(content).transpose()

        I, II = ecg[0], ecg[1]
        III = np.expand_dims(II - I, axis=0)
        aVR = np.expand_dims(-(I + II) / 2, axis=0)
        aVL = np.expand_dims(I - II / 2, axis=0)
        aVF = np.expand_dims(II - I / 2, axis=0)
        ecg = np.concatenate([ecg, III, aVR, aVL, aVF], axis=0)
        return ecg

    def __getitem__(self, index):
        sample = self.dataset[index]
        ID, ecg_dir, label = sample[0].split('.')[0], sample[2], sample[5:]
        ecg_path = os.path.join(ecg_dir, ID + '.txt')

        ecg = self.__load_ecg(ecg_path)
        ecg = (ecg - ECG_MEAN) / ECG_STD
        if self.is_train:
            ecg = self.__augment(ecg)
        return torch.FloatTensor(ecg), torch.FloatTensor(label)


class ECGLoader():

    def __init__(self, dataset, batch_size, is_train, num_threads=2):
        self.dataset = dataset
        self.is_train = is_train
        self.batch_size = batch_size
        self.num_threads = num_threads
        return

    def build(self):
        sampler, shuffle = None, True
        if self.is_train:
            sample_weights = [d[1] for d in self.dataset]
            sampler = WeightedRandomSampler(
                sample_weights, len(self.dataset)
            )
            shuffle = False

        ecg_dataset = ECGDataset(self.dataset, self.is_train)
        dataloader = DataLoader(
            ecg_dataset, batch_size=self.batch_size,
            sampler=sampler, shuffle=shuffle,
            num_workers=self.num_threads, pin_memory=False
        )
        return dataloader
