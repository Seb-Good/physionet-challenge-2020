import numpy as np


def run_12ECG_classifier(data, header_data, classes, model):

    # current_label = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # current_score = [0.9, 0.6, 0.2, 0.05, 0.2, 0.35, 0.35, 0.1, 0.1]

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    current_label[0] = 1

    return current_label, current_score


def load_12ECG_model():
    return []
