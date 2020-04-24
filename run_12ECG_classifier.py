#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data, header_data, classes, model):

    # num_classes = len(classes)
    # current_label = np.zeros(num_classes, dtype=int)
    # current_score = np.zeros(num_classes)
    #
    # # Use your classifier here to obtain a label and score for each class.
    # features=np.asarray(get_12ECG_features(data,header_data))
    # feats_reshape = features.reshape(1,-1)
    # label = model.predict(feats_reshape)
    # score = model.predict_proba(feats_reshape)
    #
    # current_label[label] = 1
    #
    # for i in range(num_classes):
    #     current_score[i] = np.array(score[0][i])

    current_label = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    current_score = [0.9, 0.6, 0.2, 0.05, 0.2, 0.35, 0.35, 0.1, 0.1]

    return current_label, current_score

def load_12ECG_model():
    return None
