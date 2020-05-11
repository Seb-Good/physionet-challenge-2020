from Decompose.SBD import *
from pykalman import KalmanFilter
from config import DATA_PATH

import pandas as pd
import numpy as np
import json
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal


class DataGenerator:
    def __init__(self, ssl):

        self.ssl = ssl
        self.target = TARGET
        self.X_train, self.y_train, self.X_test, self.X_train_ssl, self.y_train_ssl = self.load_data(
            DATA_PATH, TEST_NAME, TRAIN_NAME, SEMI_SUPERVISED, TRAIN_NEW_NAME
        )

    def load_data(self, data_path, test_name, train_name, ssl_name, train_new_name):

        # load test data
        df_test = pd.read_csv(data_path + test_name, index_col=None, header=0)
        df_test[self.target] = 1  # instead of nan

        # load train data
        df_train = pd.read_csv(data_path + train_name, index_col=None, header=0)
        # df_train = self.clean_data(df_train)

        # load ssl data
        df_train_ssl = pd.read_csv(data_path + ssl_name, index_col=None, header=0)

        # load train_new data
        # df_train_new = pd.read_csv(data_path + train_new_name, index_col=None, header=0)
        # df_train = df_train.append(df_train_new)

        # normalize signals
        # df_train, df_train_new = self.normalize(df_train, df_train_new)
        df_train, df_test = self.normalize(df_train, df_test)

        # apply initial pre-processing
        df_train, y_train = self.preprocessing_initial(df_train)
        df_test, y_test = self.preprocessing_initial(df_test)
        df_train_ssl, y_train_ssl = self.preprocessing_initial(df_train_ssl)
        # df_train_new, y_train_new = self.preprocessing_initial(df_train_new)

        # skip dataset number 8
        step = int(500000 / hparams['model']['input_size'])
        i = 7
        df_train = np.delete(df_train, np.arange(i * step, (i + 1) * step), axis=0)
        y_train = np.delete(y_train, np.arange(i * step, (i + 1) * step), axis=0)

        return df_train, y_train, df_test, df_train_ssl, y_train_ssl

    def get_train_val(self, train_ind, val_ind):

        # get trian samples
        X_train = self.X_train[train_ind]
        y_train = self.y_train[train_ind]

        # get validation samples
        X_val = self.X_train[val_ind]
        y_val = self.y_train[val_ind]

        if self.ssl:
            X_train = np.concatenate((X_train, self.X_train_ssl), axis=0)
            y_train = np.concatenate((y_train, self.y_train_ssl), axis=0)

        # X_train = np.concatenate((X_train, self.X_train_new), axis=0)
        # y_train = np.concatenate((y_train, self.y_train_new), axis=0)

        return X_train, y_train, X_val, y_val

    def preprocessing_initial(self, df):

        feature_list = df.columns.to_list()
        feature_list.remove('time')
        feature_list.remove('open_channels')
        # feature_list.remove('group')

        labels = df[self.target].values
        X = np.zeros(
            (
                int(df.shape[0] / hparams['model']['input_size']),
                hparams['model']['input_size'],
                len(feature_list),
            )
        )
        y = np.zeros((int(df.shape[0] / hparams['model']['input_size']), hparams['model']['input_size'], 1))

        for i in range(X.shape[0]):
            for n, feature in enumerate(feature_list):
                X[i, :, n] = df[feature].values[
                    i * hparams['model']['input_size'] : (i + 1) * hparams['model']['input_size']
                ]
            y[i, :, 0] = labels[
                i * hparams['model']['input_size'] : (i + 1) * hparams['model']['input_size']
            ]

        X_hist = self.get_hist(X)

        X_2 = self.squared(X)
        X_4 = self.quadratic(X)
        # X_exp = self.get_exp(X)

        X = np.concatenate((X, X_2), axis=2)
        X = np.concatenate((X, X_4), axis=2)

        X_pos, X_neg = self.add_lags(X, [1, 2, 3])

        X = np.concatenate((X, X_pos), axis=2)
        X = np.concatenate((X, X_neg), axis=2)

        X = np.concatenate((X, X_hist), axis=2)
        # X = np.concatenate((X, X_exp), axis=2)

        # X = self.add_lags(X,[1,2,3])

        return X, y

    ###################### Data Pre-processing ######################


    def get_hist(self, X):

        X_hist = np.zeros((X.shape[0], X.shape[1], 2))

        for i in range(X.shape[0]):
            h = np.histogram(X[i, :, 0], bins=4000)
            X_hist[i, :, 0] = h[1][:4000]
            X_hist[i, :, 0] = h[1][:4000]

        return X_hist



    def get_exp(self, X):
        return np.exp(X)

    def get_diff(self, X):
        diff_sig = np.diff(X, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        X = np.concatenate((X, diff_sig1), axis=2)
        return X

    def squared(self, X):
        return X ** 2

    def quadratic(self, X):
        return X ** 4

    # normalize the data (standard scaler). We can also try other scalers for a better score!
    def normalize(self, train, test):
        train_input_mean = train.signal.mean()
        train_input_sigma = train.signal.std()
        train['signal'] = (train.signal - train_input_mean) / train_input_sigma
        test['signal'] = (test.signal - train_input_mean) / train_input_sigma
        return train, test



    ###################### Feature Engineering ######################

    def add_lags(self, X, shift_list):

        if len(shift_list) == 0:
            return X
        else:

            for index, shift in enumerate(shift_list):
                if index == 0:
                    X_pos = self.shift_pos(X, shift)
                    X_neg = self.shift_neg(X, shift)
                else:
                    X_pos = np.concatenate((X_pos, self.shift_pos(X, shift)), axis=2)
                    X_neg = np.concatenate((X_neg, self.shift_neg(X, shift)), axis=2)

            return X_pos, X_neg

    def shift_pos(self, X, shift):
        X_pos = X.copy()
        X_pos[:] = 0
        X_pos[:, shift:, :] = X[:, : -1 * shift, :]
        return X_pos

    def shift_neg(self, X, shift):
        X_neg = X.copy()
        X_neg[:] = 0
        X_neg[:, : -1 * shift, :] = X[:, shift:, :]
        return X_neg


class Dataset_train(Dataset):
    def __init__(self, indexes):

        self.preprocessing = PreProcessing()

        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):

        X, y = self.load_data(idx)


        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X, y

    def load_data(self, ids, train=True):

        # load waveforms
        X = np.load(DATA_PATH + self.indexes[ids] + '.npy')
        X = self.preprocessing.run(X)

        if train:
            # load annotation
            y = np.array(json.load(open(DATA_PATH + self.indexes[ids] + '.json'))['label_train'])

            return X, y
        else:
            return X

    def get_labels(self):
        """
        :param ids: a list of ids for loading from the database
        :return: y: numpy array of labels, shape(n_samples,n_labels)
        """

        for index, i in enumerate(self.indexes):
            if index == 0:
                y = np.array(json.load(open(DATA_PATH + i + '.json'))['label_train'])
                y = np.reshape(y, (1, y.shape[0]))
            else:
                temp = np.array(json.load(open(DATA_PATH + i + '.json'))['label_train'])
                temp = np.reshape(temp, (1, temp.shape[0]))
                y = np.concatenate((y, temp), axis=0)

        return y


class Dataset_test(Dataset_train):
    def __init__(self, indexes):
        super().__init__(indexes=indexes)

    def __getitem__(self, idx):

        X = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X


class PreProcessing:
    def run(self, X):

        X = np.reshape(X, (X.shape[1], X.shape[0]))



        if X.shape[0] < 36000:
            X_new = np.zeros((36000, 12))
            X_new[: X.shape[0], :] = X[:, :]
            for i in range(12):
                X_new[X.shape[0]:, i] = X_new[X.shape[0] - 1, i]
            X = X_new
        elif X.shape[0] > 36000:
            X = X[:36000, :]
        #downsample
        #X = self.resample(X)

        return X
    #TODO: some issues with downsampling
    def resample(self, X):
        Fs = 500
        F1 = 250
        q = int(Fs / F1)
        q = int(X.shape[0]/q)

        X_dec = np.zeros((q, X.shape[1]))
        for i in range(X.shape[1]):
            X_dec[:, i] = signal.resample(X[:, i], q)#(x=X[:, i], q=q, ftype='iir')

        return X_dec

    def normalize_channels(self, X_train, X_test):

        for i in range(X_train.shape[2]):
            mean = np.mean(X_train[:, :, i])
            std = np.std(X_train[:, :, i])

            X_train[:, :, i] = (X_train[:, :, i] - mean) / std
            X_test[:, :, i] = (X_test[:, :, i] - mean) / std

        return X_train, X_test
    def apply_sbd(self, X):
        SBD_arr = SBD(X)
        return np.concatenate((X, SBD_arr), axis=2)
