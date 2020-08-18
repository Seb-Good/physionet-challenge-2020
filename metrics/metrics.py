import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


class Metric:
    def __init__(self):

        # load weights for confusion matrix
        self.weights = pd.read_csv('./metrics/weights.csv', header=0)
        self.weights = self.weights.values[:, 1:]

    # ================ Utils ================

    # Compute modified confusion matrix for multi-class, multi-label tasks.
    def compute_modified_confusion_matrix(self, labels, outputs):
        # Compute a binary multi-class, multi-label confusion matrix, where the rows
        # are the labels and the columns are the outputs.
        num_recordings, num_classes = np.shape(labels)
        A = np.zeros((num_classes, num_classes))

        # Iterate over all of the recordings.
        for i in range(num_recordings):
            # Calculate the number of positive labels and/or outputs.
            normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
            # Iterate over all of the classes.
            for j in range(num_classes):
                # Assign full and/or partial credit for each positive class.
                if labels[i, j] > 0:
                    for k in range(num_classes):
                        if outputs[i, k] > 0:
                            A[j, k] += 1.0 / normalization

        return A

    # Compute the evaluation metric for the Challenge.
    def compute(self, labels, outputs, normal_class=21):
        num_recordings, num_classes = np.shape(labels)

        # Compute the observed score.
        A = self.compute_modified_confusion_matrix(labels, outputs)
        observed_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the correct label(s).
        correct_outputs = labels
        A = self.compute_modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
        inactive_outputs[:, normal_class] = 1
        A = self.compute_modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(self.weights * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score

