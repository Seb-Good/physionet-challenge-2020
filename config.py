import ast
import os
import torch

# names:
DATA_PATH = './data/processed/'
TRAIN_NAME = 'train_clean_kalman.csv'
TEST_NAME = 'test_clean_kalman.csv'
SEMI_SUPERVISED = 'semi_supervised_0.csv'
TRAIN_NEW_NAME = 'train_new.csv'

TARGET = 'open_channels'


PIC_FOLDER = './data/pictures/'
DEBUG_FOLDER = './data/CV_debug/'

for f in [PIC_FOLDER, DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)


with open('class_weights.txt', 'r') as f:
    s = f.read()
    CLASS_WEIGHTS = ast.literal_eval(s)


# fix random seed
