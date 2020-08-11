# basic libs
import numpy as np
import json
import os

# pytorch
import torch
from torch.utils.data import Dataset

np.random.seed(42)



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

        if self.patients[id][0] == 'A':
            data_folder = 'A'
        elif self.patients[id][0] == 'Q':
            data_folder = 'B'
        elif self.patients[id][0] == 'I':
            data_folder = 'C'
        elif self.patients[id][0] == 'S':
            data_folder = 'D'
        elif self.patients[id][0] == 'H':
            data_folder = 'E'
        elif self.patients[id][0] == 'E':
            data_folder = 'F'
        else:
            print(1)

        data_folder = f'./data/{data_folder}/formatted/'

        # load waveforms
        X = np.load(data_folder+self.patients[id] + '.npy')

        # load annotation
        y = json.load(open(data_folder + self.patients[id] + '.json'))
        X = X /y['amp_conversion']

        return X, y['labels_training_merged']

        # if train:
        #     # load annotation
        #     y = json.load(open(data_folder+self.patients[id] + '.json'))
        #
        #     return X, y['labels_training_merged']
        # else:
        #     return X

    def get_labels(self):
        """
        :param ids: a list of ids for loading from the database
        :return: y: numpy array of labels, shape(n_samples,n_labels)
        """

        for index, record in enumerate(self.patients):

            if record[0] == 'A':
                data_folder = 'A'

            elif record[0] == 'Q':
                data_folder = 'B'

            elif record[0] == 'I':
                data_folder = 'C'

            elif record[0] == 'S':
                data_folder = 'D'

            elif record[0] == 'H':
                data_folder = 'E'

            elif record[0] == 'E':
                data_folder = 'F'

            data_folder = f'./data/{data_folder}/formatted/'


            if index == 0:
                y = np.array([json.load(open(data_folder+record + '.json'))['labels_training_merged']])
                y = np.reshape(y, (1, 27))
            else:
                temp = np.array([json.load(open(data_folder+record + '.json'))['labels_training_merged']])
                temp = np.reshape(temp, (1, 27))
                y = np.concatenate((y, temp), axis=0)

        return y

    def my_collate(self,batch):
        """
        This function was created to handle a variable-length of the
        :param batch: tuple(data,target)
        :return: list[data_tensor(batch_size,n_samples_channels), target_tensor(batch_size,n_classes)]
        """
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]

        # define the max size of the batch
        m_size = 0
        for element in data:
            if m_size < element.shape[0]:
                m_size = element.shape[0]

        # zero pooling
        for index, element in enumerate(data):
            if m_size > element.shape[0]:
                padding = np.zeros((m_size-element.shape[0], element.shape[1]))
                padding = torch.from_numpy(padding)
                data[index] = torch.cat([element, padding], dim=0)
                padding = padding.detach()

        data = torch.stack(data)
        target = torch.stack(target)

        return [data, target]


class Dataset_test(Dataset_train):
    def __init__(self, patients):
        super().__init__(patients=patients)

    def __getitem__(self, idx):

        X,y = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X