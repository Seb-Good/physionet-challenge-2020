# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import os

from data_generator import Dataset_train, Dataset_test
from metrics import Metric


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(
        self, hparams, split_table_path, split_table_name, pic_folder, debug_folder,
    ):

        # load the model

        self.hparams = hparams

        print('\n')
        print('Selected Learning rate:', self.hparams['lr'])
        print('\n')

        self.pic_folder = pic_folder
        self.debug_folder = debug_folder
        self.split_table_path = split_table_path
        self.split_table_name = split_table_name

        self.splits = self.load_split_table()

    def load_split_table(self):

        splits = []

        for i in range(5):
            data = json.load(open(self.split_table_path + str(i) + self.split_table_name))
            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold != self.hparams['start_fold']:
                    continue

            train = Dataset_train(indexes=self.splits['train'].values[fold])
            valid = Dataset_train(indexes=self.splits['val'].values[fold])

            X, y = train.__getitem__(0)

            # TODO: model will require mutiple hparams
            self.model = self.model(input_size=X.shape[1], n_channels=X.shape[2], hparams=self.hparams,)

            # train model
            self.model.fit(train=train, valid=valid)

            # get model predictions
            valid = Dataset_test(indexes=self.splits['val'].values[fold])
            pred_val = self.model.predict(valid)
            heatmap = self.model.get_heatmap(valid)

            y_val = valid.get_labels()
            fold_score = metric(y_val, pred_val)

            # save the model
            self.model.model_save(
                self.hparams['model_path']
                + self.hparams['model_name']
                + '_'
                + str(fold)
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )

            # create a dictionary for debugging
            debugging = {}
            for index, i in enumerate(self.splits['val'].values[fold]):
                temp = {}
                temp['heatmap'] = np.abs(heatmap[index, :]).tolist()
                temp['labels'] = y_val[index, :].tolist()
                temp['preds'] = pred_val[index, :].tolist()
                debugging[f'{i}'] = temp

            # save debug data
            with open(self.debug_folder + str(fold) + '_fold_' + str(fold_score) + '.txt', "w",) as file:
                file.write(str(debugging))
                file.close()

        return fold_score
