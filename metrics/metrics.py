import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Metric():

    def __init__(self):

        #load weights for confusion matrix
        self.weights = pd.read_csv('./metrics/weights.csv', header=0)
        self.weights = self.weights.values[:, 1:]


    #================ Utils ================
    # Compute modified confusion matrix for multi-class, multi-label tasks.
    def compute_modified_confusion_matrix(self,labels, outputs):
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

    def find_opt_thresold(self,labels, outputs):

        threshold_grid = np.arange(0.01,0.99,0.05).tolist()
        threshold_opt = np.zeros((27))



        #TODO
        print('Finding the optimal threshold')
        for i in tqdm(range(27)):
            outputs_thresholded = np.zeros((outputs.shape[0], outputs.shape[1]))
            scores = []
            for threshold in threshold_grid:
                outputs_thresholded[:,i] = outputs[:,i]
                outputs_thresholded[np.where(outputs_thresholded >= threshold)] = 1
                outputs_thresholded[np.where(outputs_thresholded < threshold)] = 0
                scores.append(self.compute(labels,outputs_thresholded))
            scores = np.array(scores)
            threshold_opt[i] = threshold_grid[np.where(scores == np.max(scores))]

        for i in range(27):
            output = outputs[:, i]
            output[np.where(output) >= threshold_opt[i]] = 1
            output[np.where(output) < threshold_opt[i]] = 0
            outputs[:, i] = output
        #save thresholds


        return labels, outputs


