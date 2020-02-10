"""
training_data.py
----------------
This module provides classes and methods for creating a training dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import json


# Local imports
from src import DATA_PATH, DATA_FILE_NAME, EXTRACTED_FOLDER_NAME, LABELS_LOOKUP, FS


class TrainingData(object):

    def __init__(self):

        # Set attributes
        self.formatted_path = os.path.join(DATA_PATH, 'formatted')
        self.training_path = os.path.join(DATA_PATH, 'training')

