# Import 3rd party libraries
import os

# Set working directory
WORKING_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Set data directory
DATA_PATH = os.path.join(WORKING_PATH, 'data')

# Set output directory
OUTPUT_PATH = os.path.join(WORKING_PATH, 'output')

# Set weights directory
WEIGHTS_PATH = os.path.join(WORKING_PATH, 'kardioml', 'scoring', 'weights.csv')

# Dataset file name
DATA_FILE_NAMES = {
    'A': 'PhysioNetChallenge2020_Training_CPSC.tar.gz',
    'B': 'PhysioNetChallenge2020_Training_2.tar.gz',
    'C': 'PhysioNetChallenge2020_Training_StPetersburg.tar.gz',
    'D': 'PhysioNetChallenge2020_Training_PTB.tar.gz',
    'E': 'PhysioNetChallenge2020_PTB-XL.tar.gz',
    'F': 'PhysioNetChallenge2020_Training_E.tar.gz',
}

# Extracted folder name
EXTRACTED_FOLDER_NAMES = {
    'A': 'Training_WFDB',
    'B': 'Training_2',
    'C': 'Training_StPetersburg',
    'D': 'Training_PTB',
    'E': 'WFDB',
    'F': 'WFDB',
}

# ECG leads
ECG_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Filter band limits
FILTER_BAND_LIMITS = (3, 45)

# Number of leads
NUM_LEADS = 12

# Number of labels
LABELS_COUNT = 27
