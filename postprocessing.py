import numpy as np

class PostProcessing():


    def __init__(self):

        self.threshold = 0.5


    def run(self,predictions):

        predictions[np.where(predictions >= self.threshold)] = 1
        predictions[np.where(predictions < self.threshold)] = 0

        return predictions