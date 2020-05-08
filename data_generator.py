from Decompose.SBD import *
from pykalman import KalmanFilter
from config import DATA_PATH, TEST_NAME, TRAIN_NAME, TARGET, SEMI_SUPERVISED,TRAIN_NEW_NAME
from model.wavenet import hparams
import pandas as pd
import numpy as np
import gc


class DataGenerator:
    def __init__(self,ssl):

        self.ssl = ssl
        self.target = TARGET
        self.X_train, self.y_train, self.X_test,self.X_train_ssl,self.y_train_ssl = self.load_data(DATA_PATH, TEST_NAME, TRAIN_NAME,SEMI_SUPERVISED,TRAIN_NEW_NAME)

    def load_data(self, data_path, test_name, train_name,ssl_name,train_new_name):

        # load test data
        df_test = pd.read_csv(data_path + test_name, index_col=None, header=0)
        df_test[self.target] = 1  # instead of nan

        # load train data
        df_train = pd.read_csv(data_path + train_name, index_col=None, header=0)
        #df_train = self.clean_data(df_train)

        # load ssl data
        df_train_ssl = pd.read_csv(data_path + ssl_name, index_col=None, header=0)

        # load train_new data
        #df_train_new = pd.read_csv(data_path + train_new_name, index_col=None, header=0)
        #df_train = df_train.append(df_train_new)

        # normalize signals
        #df_train, df_train_new = self.normalize(df_train, df_train_new)
        df_train, df_test = self.normalize(df_train, df_test)

        # apply initial pre-processing
        df_train, y_train = self.preprocessing_initial(df_train)
        df_test, y_test = self.preprocessing_initial(df_test)
        df_train_ssl, y_train_ssl = self.preprocessing_initial(df_train_ssl)
        #df_train_new, y_train_new = self.preprocessing_initial(df_train_new)


        #skip dataset number 8
        step = int(500000/hparams['model']['input_size'])
        i = 7
        df_train = np.delete(df_train,np.arange(i*step,(i+1)*step),axis=0)
        y_train = np.delete(y_train, np.arange(i * step, (i + 1) * step), axis=0)

        return df_train, y_train, df_test,df_train_ssl,y_train_ssl

    def get_train_val(self, train_ind, val_ind):

        # get trian samples
        X_train = self.X_train[train_ind]
        y_train = self.y_train[train_ind]

        # get validation samples
        X_val = self.X_train[val_ind]
        y_val = self.y_train[val_ind]

        if self.ssl:
            X_train  = np.concatenate((X_train, self.X_train_ssl), axis=0)
            y_train  = np.concatenate((y_train, self.y_train_ssl), axis=0)

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
        #X_exp = self.get_exp(X)

        X = np.concatenate((X, X_2), axis=2)
        X = np.concatenate((X, X_4), axis=2)

        X_pos,X_neg = self.add_lags(X,[1,2,3])

        X = np.concatenate((X, X_pos), axis=2)
        X = np.concatenate((X, X_neg), axis=2)

        X = np.concatenate((X, X_hist), axis=2)
        #X = np.concatenate((X, X_exp), axis=2)

        # X = self.add_lags(X,[1,2,3])

        return X, y

    ###################### Data Pre-processing ######################

    def clean_data(self,df_train):

        th_0_1 = -2.24
        th_1_2 = -0.9
        th_2_3 = 0.375
        th_7_8 = 3.9
        th_8_9 = 5
        th_9_10 = 6.4

        labels = df_train['open_channels'].values

        labels[np.where(df_train['signal'].values <= th_0_1)] = 0

        labels[np.where((df_train['signal'].values > th_0_1) & (df_train['signal'].values <= th_1_2))] = 1

        labels[np.where((df_train['signal'].values > th_1_2) & (df_train['signal'].values <= th_2_3))] = 2

        labels[np.where((df_train['signal'].values > th_7_8) & (df_train['signal'].values <= th_8_9))] = 8

        labels[np.where((df_train['signal'].values > th_8_9) & (df_train['signal'].values <= th_9_10))] = 9

        labels[np.where((df_train['signal'].values > th_9_10))] = 10

        df_train['open_channels'] = labels

        return df_train

    def get_hist(self, X):

        X_hist = np.zeros((X.shape[0],X.shape[1],2))

        for i in range(X.shape[0]):
            h = np.histogram(X[i,:,0], bins=4000)
            X_hist[i,:,0] = h[1][:4000]
            X_hist[i, :, 0] = h[1][:4000]

        return X_hist

    def apply_sbd(self, X):
        SBD_arr = SBD(X)
        return np.concatenate((X, SBD_arr), axis=2)

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

    def normalize_channels(self,X_train,X_test):

        for i in range(X_train.shape[2]):
            mean = np.mean(X_train[:,:,i])
            std = np.std(X_train[:, :, i])

            X_train[:, :, i] = (X_train[:,:,i] - mean)/std
            X_test[:, :, i] = (X_test[:, :, i] - mean) / std

        return X_train,X_test

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

            return X_pos,X_neg

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
