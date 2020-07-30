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
    def __init__(self, input_folders, output_folder, split_folder):

        self.input_folders = input_folders
        self.output_folder = output_folder
        self.split_folder = split_folder


    def run(self):

        # download a json file for exclusions
        exclude = json.load(open(self.split_folder + 'exclude.json'))

        # get a list of patients
        self.patients = []
        for input_folder in self.input_folders:
            for patient in os.listdir(input_folder):
                if patient in exclude.items():
                    continue
                else:
                    self.patients.append(patient)


        # split data into folds
        self.split_table = self.create_split_table()

        print('Total number of patients: ', len([*self.patients]))

        return 0

    def create_split_table(self):

        #TODO: finish up cross-validation loop
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        split_table = []

        for index, (train, val) in enumerate(kfold.split([*self.patients])):
            split = {}
            temp = self.patients.keys()
            split['train'] = np.array([*self.patients])[train].tolist()
            split['val'] = np.array([*self.patients])[val].tolist()

            split_table.append(split)
            with open(self.split_folder + str(index) + '_fold.json', 'w') as outfile:
                json.dump(split, outfile)

        # Generate DataFrame
        split_table = pd.DataFrame(split_table)

        return split_table