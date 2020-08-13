# basic libs
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
import os
import gc
from tqdm import tqdm
from shutil import rmtree
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample

# pytorch
from torch import nn
import torch


class PrepareData:
    def __init__(self, input_folders, split_folder, split_table_name):

        self.input_folders = input_folders
        self.split_folder = split_folder
        self.split_table_name = split_table_name

    def run(self):

        # download a json file for exclusions
        exclude = json.load(open(self.split_folder + 'exclude.json'))

        # get a list of patients
        self.patients = []
        for input_folder in self.input_folders:
            for patient in [i for i in os.listdir(input_folder) if i.find('.npy') != -1]:
                if patient[:-4] in exclude.items():
                    continue
                else:
                    self.patients.append(patient)

        # split data into folds

        self.split_table = self.create_split_table()
        print('Total number of patients: ', len(self.patients))
        return 0

    def load_labels(self, name):
        y = json.load(open(name + '.json'))
        return y

    def create_split_table(self):

        kfold = KFold(n_splits=6, random_state=42, shuffle=True)

        split_table = []

        for index, (train, val) in enumerate(kfold.split(self.input_folders)):
            split = {}
            train = [i for index, i in enumerate(self.input_folders) if index in train.tolist()]
            val = [i for index, i in enumerate(self.input_folders) if index in val.tolist()]

            patients_train = []
            patients_val = []

            for dataset in train:
                patients = [i for i in os.listdir(dataset) if i.find('.npy') != -1]
                print(f'Start checking each patient in dataset {dataset[7]}... for training')
                for patient in tqdm(patients):
                    y = self.load_labels(dataset + patient[:-4])
                    if y['labels_training_merged'] == None:
                        continue
                    patients_train.append(dataset + patient[:-4])

            for dataset in val:
                patients = [i for i in os.listdir(dataset) if i.find('.npy') != -1]
                print(f'Start checking each patient in dataset {dataset[7]}... for validation')
                for patient in patients:
                    y = self.load_labels(dataset + patient[:-4])
                    if y['labels_training_merged'] == None:
                        continue
                    patients_val.append(dataset + patient[:-4])

            split['train'] = patients_train
            split['val'] = patients_val

            split_table.append(split)
            with open(self.split_folder + str(index) + '_fold.json', 'w') as outfile:
                json.dump(split, outfile)

        # Generate DataFrame
        split_table = pd.DataFrame(split_table)

        return split_table
