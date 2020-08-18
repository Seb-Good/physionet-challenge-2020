import numpy as np
import torch

# from pytorch_toolbelt import losses as L


class EarlyStopping:
    def __init__(
        self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True,
    ):
        self.patience, self.delta, self.checkpoint_path = (
            patience,
            delta,
            checkpoint_path,
        )
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

        self.threshold = None

    def load_best_weights(self):
        return torch.load(self.checkpoint_path)

    def __call__(self, score, model,threshold):

        if self.is_maximize:
            if self.best_score is None or (score - self.delta > self.best_score):
                torch.save(model,self.checkpoint_path)
                self.best_score, self.counter = score, 0
                self.threshold = threshold
                return 1
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return 2
        else:
            if self.best_score is None or (score + self.delta < self.best_score):
                torch.save(model,self.checkpoint_path)
                self.best_score, self.counter = score, 0
                self.threshold = threshold
                return 1
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return 2