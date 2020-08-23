import ast
import os

# select the type of the model here
from models.ecgnet import Model, hparams

# names:
DATA_PATH = []
for dataset in ['A', 'B', 'C', 'D', 'E', 'F']:
    DATA_PATH.append('./data/scipy_resample_1000_hz/' + dataset + '/formatted/')

SPLIT_TABLE_PATH = './'
SPLIT_TABLE_NAME = 'cv.json'

PIC_FOLDER = './data/pictures/'
DEBUG_FOLDER = './data/CV_debug/'

for f in [PIC_FOLDER, DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)
