# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from model.config_ecgnet import hparams
from data_generator import Dataset_train, Dataset_test
from model.ecgnet_v2 import DL_model
from config import SPLIT_TABLE_PATH, PIC_FOLDER, DEBUG_FOLDER
import torch
import os
from metrics import compute_beta_score


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class Pipeline:
    def __init__(
        self,
        start_fold,
        epochs,
        batch_size,
        lr,
        split_table_path=SPLIT_TABLE_PATH,
        pic_folder=PIC_FOLDER,
        debug_folder=DEBUG_FOLDER,
    ):

        # load the model
        self.start_fold = start_fold

        hparams['batch_size'] = batch_size
        hparams['epochs'] = epochs
        hparams['lr'] = lr

        print('\n')
        print('Selected Learning rate:', hparams['lr'])
        print('\n')

        self.pic_folder = pic_folder
        self.debug_folder = debug_folder
        self.split_table_path = split_table_path

        self.splits = self.load_split_table()

    def load_split_table(self):

        splits = []

        for i in range(5):
            data = json.load(open(self.split_table_path + f'training_lookup_cv{i+1}_multilabel.json'))
            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold != self.start_fold:
                continue

            train = Dataset_train(indexes=self.splits['train'].values[fold])
            valid = Dataset_train(indexes=self.splits['val'].values[fold])

            X, y = train.__getitem__(0)

            self.model = DL_model(n_channels=X.shape[1])

            # train model
            self.model.fit(train=train, valid=valid)

            # get model predictions
            valid = Dataset_test(indexes=self.splits['val'].values[fold])
            pred_val = self.model.predict(valid)
            heatmap = self.model.get_heatmap(valid)
            pred_val_raw = pred_val.copy()
            pred_val = self.model.apply_threshold(pred_val)
            y_val = valid.get_labels()
            f2_val, g2_val = compute_beta_score(y_val, pred_val, beta=2, num_classes=9)

            fold_score = np.sqrt(f2_val * g2_val)

            # save the model
            self.model.model_save(
                hparams['model_path']
                + hparams['model_name']
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
                temp['preds_raw'] = pred_val_raw[index, :].tolist()
                debugging[f'{i}'] = temp

            # save debug data
            with open(self.debug_folder + str(fold) + '_fold_' + str(fold_score) + '.txt', "w") as file:
                file.write(str(debugging))
                file.close()

        return fold_score
