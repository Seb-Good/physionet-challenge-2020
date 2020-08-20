from functools import partial

import numpy as np
from tqdm import tqdm
from metrics import Metric
import time
from concurrent.futures import ProcessPoolExecutor



class PostProcessing():


    def __init__(self,fold):

        self.fold = fold

        self.threshold = float(open(f"threshold_{self.fold}.txt", "r").read())#0.5#0.1
        self.metric = Metric()


    def run(self,predictions):

        predictions_processed = predictions.copy()

        #if somth is found, its not a normal
        predictions_processed[np.where(predictions_processed >= self.threshold)] = 1
        predictions_processed[np.where(predictions_processed < self.threshold)] = 0

        return predictions_processed

    def find_opt_thresold(self, labels, outputs):

        threshold_grid = np.arange(0.05, 0.99, 0.05).tolist()
        threshold_opt = np.zeros((27))

        unit_threshold= partial(self._unit_threshold,labels=labels,outputs=outputs)

        start = time.time()
        with ProcessPoolExecutor(max_workers=20) as pool:
            result = pool.map(
                 unit_threshold,threshold_grid
        )
        scores = list(result)
        print(f'Processing time: {(time.time() - start)/60}')

        # print('Finding the optimal threshold')
        # for threshold in tqdm(threshold_grid):
        #
        #     predictions = outputs.copy()
        #
        #     predictions[np.where(predictions >= threshold)] = 1
        #     predictions[np.where(predictions < threshold)] = 0
        #
        #     scores.append(self.metric.compute(labels, predictions))

        scores = np.array(scores)
        a = np.where(scores == np.max(scores))
        if len(a)>1:
            a = [0]
            threshold_opt = threshold_grid[a[0]]
        else:
            threshold_opt = threshold_grid[a[0][0]]

        return threshold_opt

    def _unit_threshold(self,threshold,labels,outputs):

        predictions = outputs.copy()

        predictions[np.where(predictions >= threshold)] = 1
        predictions[np.where(predictions < threshold)] = 0

        return self.metric.compute(labels, predictions)

    def update_threshold(self,threshold):
        f = open(f"threshold_{self.fold}.txt", "w")
        f.write(str(threshold))
        f.close()
        self.threshold = threshold

        # # TODO
        # print('Finding the optimal threshold')
        # for i in tqdm(range(27)):
        #     outputs_thresholded = np.zeros((outputs.shape[0], outputs.shape[1]))+1
        #     scores = []
        #     for threshold in threshold_grid:
        #         outputs_thresholded[:, i] = outputs[:, i]
        #         outputs_thresholded[np.where(outputs_thresholded >= threshold)] = 1
        #         outputs_thresholded[np.where(outputs_thresholded < threshold)] = 0
        #         scores.append(self.metric.compute(labels, outputs_thresholded))
        #     scores = np.array(scores)
        #     a = np.where(scores == np.max(scores))
        #     print(a)
        #     if len(a)>1:
        #         a = [0]
        #         threshold_opt[i] = threshold_grid[a[0]]
        #     else:
        #         threshold_opt[i] = threshold_grid[a[0][0]]
        #
        # for i in range(27):
        #     output = outputs[:, i]
        #     output[np.where(output >= threshold_opt[i])] = 1
        #     output[np.where(output < threshold_opt[i])] = 0
        #     outputs[:, i] = output
        # # save thresholds

        #return labels, outputs

