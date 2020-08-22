# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import os
from tqdm import tqdm

from data_generator import Dataset_train, Dataset_test
from metrics import Metric
from postprocessing import PostProcessing

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(self, hparams, split_table_path, split_table_name, debug_folder, model, gpu,downsample):

        # load the model

        self.hparams = hparams
        self.model = model
        self.gpu = gpu
        self.downsample = downsample

        print('\n')
        print('Selected Learning rate:', self.hparams['lr'])
        print('\n')

        self.debug_folder = debug_folder
        self.split_table_path = split_table_path
        self.split_table_name = split_table_name
        self.exclusions = ['S0431',
                           'S0326'
                           'S0453'
                           'S0458'
                           'A5766'
                           'A0227'
                           'A0238'
                           'A1516'
                           'A5179'
                           'Q1807'
                           'Q3568'
                           'E10256'
                           'E07341'
                           'E05758']


        self.splits = self.load_split_table()
        self.metric = Metric()



    def load_split_table(self):

        splits = []

        split_files = [i for i in os.listdir(self.split_table_path) if i.find('fold') != -1]

        for i in range(len(split_files)):
            data = json.load(open(self.split_table_path + str(i) + '_' + self.split_table_name))

            train_data = data['train']
            for index, i in enumerate(train_data):
                i = i.split('\\')
                i = i[-1]
                train_data[index] = i

            val_data = data['val']
            for index, i in enumerate(val_data):
                i = i.split('\\')
                i = i[-1]
                val_data[index] = i

            dataset_train = []
            for i in train_data:
                if i in self.exclusions:
                    continue
                if i[0] != 'Q' and i[0] != 'S' and i[0] != 'A' and i[0] != 'H' and i[0] != 'E':  # A, B , D, E datasets
                    continue
                dataset_train.append(i)

            dataset_val = []
            for i in val_data:
                if i in self.exclusions:
                    continue
                if i[0] != 'Q' and i[0] != 'S' and i[0] != 'A' and i[0] != 'H' and i[0] != 'E':  # A, B , D, E datasets
                    continue
                dataset_val.append(i)

            data['train'] = dataset_train#+self.additinal_data
            data['val'] = dataset_val

            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold != self.hparams['start_fold']:
                    continue
            #TODO
            train = Dataset_train(self.splits['train'].values[fold], aug=False,downsample=self.downsample)
            valid = Dataset_train(self.splits['val'].values[fold], aug=False,downsample=self.downsample)

            X, y = train.__getitem__(0)

            self.model = self.model(
                input_size=X.shape[0], n_channels=X.shape[1], hparams=self.hparams, gpu=self.gpu
            )

            # train model
            self.model.fit(train=train, valid=valid)

            # get model predictions
            y_val,pred_val = self.model.predict(valid)
            self.postprocessing = PostProcessing(fold=self.hparams['start_fold'])

            pred_val_processed = self.postprocessing.run(pred_val)

            # TODO: add activations
            # heatmap = self.model.get_heatmap(valid)


            fold_score = self.metric.compute(y_val, pred_val_processed)
            print("Model's final scrore: ",fold_score)
            # save the model
            self.model.model_save(
                self.hparams['model_path']
                + self.hparams['model_name']+f"_{self.hparams['start_fold']}"
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )


            # create a dictionary for debugging
            self.save_debug_data(pred_val, self.splits['val'].values[fold])



        return fold_score

    def save_debug_data(self, pred_val, validation_list):

        for index, data in enumerate(validation_list):

            if data[0] == 'A':
                data_folder = 'A'

            elif data[0] == 'Q':
                data_folder = 'B'

            elif data[0] == 'I':
                data_folder = 'C'

            elif data[0] == 'S':
                data_folder = 'D'

            elif data[0] == 'H':
                data_folder = 'E'

            elif data[0] == 'E':
                data_folder = 'F'

            data_folder = f'./data/CV_debug/{data_folder}/'

            prediction = {}
            prediction['predicted_label'] = pred_val[index].tolist()
            # save debug data
            with open(data_folder + data + '.json', 'w') as outfile:
                json.dump(prediction, outfile)

        return True
