"""
components:
1. model
2. postpocessing:
a) thresholds for classes
b) averaging strategy: majority voting/averaging os probabilities
"""
import numpy as np

from kardioml.data.resample import Resampling
from postprocessing import PostProcessing
from data_generator import Dataset_train
from config import Model


class Predict():

    def __init__(self):


        #load the model
        self.model = Model()
        self.model.model_load("./inference_models/ecgnet_0_fold_0.6078759902401878.pt")

        #load preprocessng pipeline

        #load thresholds
        self.postptocessing = PostProcessing(0)

        # threshold = 0
        # for fold in range(6):
        #     threshold += float(open(f"threshold_{fold}.txt", "r").read())/6
        #
        # self.postptocessing.threshold = threshold


    def predict(self,signal,meta):

        ############## Preprocessing ##############
        #downsampling
        X_resampled = np.zeros((signal.shape[0] // 2, 12))
        for i in range(12):
            X_resampled[:, i] = self.resampling.downsample(signal[:, 0], order=2)

        #apply preprocessing
        signal = self.apply_amplitude_scaling(X=X_resampled,y=meta)

        # padding
        sig_length = 19000

        if X_resampled.shape[0] < sig_length:
            padding = np.zeros((sig_length - X_resampled.shape[0], X_resampled.shape[1]))
            X = np.concatenate([X_resampled, padding], axis=0)
        if X_resampled.shape[0] > sig_length:
            X_resampled = X_resampled[:sig_length, :]

        ############## Predictions ##############
        predict = self.model.predict(X_resampled)

        ############## Postprocessing ##############

        predict = self.postptocessing.run(predict)
        predict = list(predict)

        if predict[4] > 0 or predict[18] > 0:
            predict[4] = 1
            predict[18] = 1
        if predict[23] > 0 or predict[12] > 0:
            predict[23] = 1
            predict[12] = 1
        if predict[26] > 0 or predict[13] > 0:
            predict[26] = 1
            predict[13] = 1

        return predict

    @staticmethod
    def apply_amplitude_scaling(X, y):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if y['rpeaks']:
            for channel_rpeaks in y['rpeaks']:
                if channel_rpeaks:
                    return X / np.median(X[y['rpeaks'][0], 0])
        return (X - X.mean()) / X.std()



