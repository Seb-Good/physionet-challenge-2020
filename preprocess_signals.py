import pandas as pd
import numpy as np
import json
import scipy.signal as signal
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
import matplotlib.pyplot as plt
from tqdm import tqdm

from Decompose.SBD import SBD


class PreProcessing:
    def run(self, X):

        # pre-processing
        X = np.reshape(X, (-1, 12))

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

        # remove baseline wander
        # X = self.baseline_compensation(X)

        # feature engineering
        R_peaks = self.apply_r_peaks_segmentation(X)

        X = self.normalize_channels(X, R_peaks)

        # X = np.concatenate((X, R_peaks.reshape(-1, 1)), axis=1)

        X = self.apply_sbd(X).astype(np.float32)
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

    def baseline_compensation(self, X):

        win_length = 501

        # raw = X[:, 0].copy()

        for i in range(X.shape[1]):
            channel = X[:, i].copy()
            channel = np.concatenate((np.zeros((win_length - 1)), channel))
            channel = np.concatenate((channel, np.zeros((int(win_length / 2)))))
            channel = pd.Series(channel).rolling(win_length).median().values
            X[:, i] -= channel[win_length - 1 + int(win_length / 2) :]

        # plt.plot( X[:1000,0])
        # plt.plot(raw[:1000])
        # plt.show()
        return X

    def normalize_channels(self, X, peaks):

        peaks = np.where(peaks > 0)[0]
        scaling_val = np.median(X[peaks, 0])

        for i in range(X.shape[1]):
            X[:, i] = X[:, i] / scaling_val

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


def load_data(filename):

    # load waveforms
    X = np.load('./data/formatted/' + filename + '.npy')

    return X


def save_data(filename, X):

    # load waveforms
    np.save('./data/processed/' + filename + '.npy', X)

    return True


def load_split_table():

    records = []

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    train_prev = []

    for i in range(5):
        data = json.load(open('./data/fold_split/' + f'training_lookup_cv{i+1}.json'))
        for i in data['val']:
            records.append(i)

    return records


def main():

    # load a list of files
    records = load_split_table()

    # object for processing
    processing = PreProcessing()

    for i in tqdm(records):
        signal = load_data(i)
        signal = processing.run(signal)
        save_data(i, signal)


if __name__ == "__main__":
    main()
