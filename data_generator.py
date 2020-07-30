from Decompose.SBD import *
from config import DATA_PATH, LABELS_PATH

import pandas as pd
import numpy as np
import json
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
import matplotlib.pyplot as plt


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
        # X = self.preprocessing.run(X)

        if train:
            # load annotation
            y = np.array(json.load(open(LABELS_PATH + self.indexes[ids] + '.json'))['label_train'])

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
                y = np.array(json.load(open(LABELS_PATH + i + '.json'))['label_train'])
                y = np.reshape(y, (1, y.shape[0]))
            else:
                temp = np.array(json.load(open(LABELS_PATH + i + '.json'))['label_train'])
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

        # pre-processing
        X = X.T  # np.reshape(X, (X.shape[1], X.shape[0]))

        # normalize signals
        # X = self.normalize_channels(X)

        if X.shape[0] < 36000:
            X_new = np.zeros((36000, 12))
            X_new[: X.shape[0], :] = X[:, :]
            for i in range(12):
                X_new[X.shape[0] :, i] = X_new[X.shape[0] - 1, i]
            X = X_new
        elif X.shape[0] > 36000:
            X = X[:36000, :]

        # downsample
        X = self.resample(X)

        # feature engineering
        # R_peaks = self.apply_r_peaks_segmentation(X)
        # X = np.concatenate((X,R_peaks.reshape(-1,1)),axis=1)
        # X = self.apply_sbd(X).astype(np.float32)
        return X

    ######################## Signal pre-processing #####################
    def resample(self, X):
        Fs = 500
        F1 = 400
        q = Fs / F1
        q = int(X.shape[0] / q)

        X_dec = np.zeros((q, X.shape[1]))
        for i in range(X.shape[1]):
            X_dec[:, i] = signal.resample(X[:, i], q)  # (x=X[:, i], q=q, ftype='iir')

        return X_dec

    def normalize_channels(self, X):

        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

        return X

    def apply_sbd(self, X):
        X = np.reshape(X, (1, X.shape[0], -1))
        for i in range(1):
            SBD_arr = SBD(X[:, :, i])
            X = np.concatenate((X, SBD_arr), axis=2)
        X = np.reshape(X, (X.shape[1], X.shape[2]))
        return X

    def apply_r_peaks_segmentation(self, X):

        peaks = self.find_peaks(X)

        R_peaks = np.zeros((X.shape[0]))

        R_peaks[peaks] = 1

        return R_peaks

    ######################## Features #####################

    ######################## Functions #####################
    def find_peaks(self, X, channel=0):

        X_processed = X[:, channel].copy()
        X_processed = filter_signal(
            signal=X_processed, ftype='FIR', band='bandpass', order=150, frequency=[3, 45], sampling_rate=400
        )[0]

        # check original polarity
        ecg_object = ecg.ecg(signal=X_processed, sampling_rate=400, show=False)
        peaks_plus = ecg_object['rpeaks']  # rpeak indices

        # check reversed polarity
        ecg_object = ecg.ecg(signal=-1 * X_processed, sampling_rate=400, show=False)
        peaks_minus = ecg_object['rpeaks']  #

        # select polarity
        if np.abs(np.median(X_processed[peaks_minus])) > np.abs(np.median(X_processed[peaks_plus])):
            peaks = peaks_minus.copy()
        else:
            peaks = peaks_plus.copy()

        return peaks
