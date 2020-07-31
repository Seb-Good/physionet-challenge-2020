# basic libs
import numpy as np
import json
import os

# pytorch
import torch
from torch.utils.data import Dataset

np.random.seed(42)


# TODO: need to refactor to upload large datasets in batches
class Dataset_train(Dataset):
    def __init__(self, patients):

        self.patients = patients



    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        X, y = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X, y

    def load_data(self, id, train=True):

        # load waveforms
        X = np.load(self.patients[id] + '.npy')
        # X = self.preprocessing.run(X)

        if train:
            # load annotation
            y = np.array(json.load(open(self.patients[id] + '.json'))['labels_training_merged'])

            return X, y
        else:
            return X

    def get_labels(self):
        """
        :param ids: a list of ids for loading from the database
        :return: y: numpy array of labels, shape(n_samples,n_labels)
        """

        for index, record in enumerate(self.patients):
            if index == 0:
                y = np.array([json.load(open(record + '.json'))['label']])
                y = np.reshape(y, (1, 1))
            else:
                temp = np.array([json.load(open(record + '.json'))['label']])
                temp = np.reshape(temp, (1, 1))
                y = np.concatenate((y, temp), axis=0)

        return y


class Dataset_test(Dataset_train):
    def __init__(self, patients):
        super().__init__(patients=patients)

    def __getitem__(self, idx):

        X = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X