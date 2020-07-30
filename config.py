import ast
import os


# names:
DATA_PATH = './data/processed/'
LABELS_PATH = './data/formatted/'
SPLIT_TABLE_PATH = './data/fold_split/'


PIC_FOLDER = './data/pictures/'
DEBUG_FOLDER = './data/CV_debug/'

for f in [PIC_FOLDER, DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)

# load class weights
# with open('class_weights.txt', 'r') as f:
#     s = f.read()
#     CLASS_WEIGHTS = ast.literal_eval(s)
