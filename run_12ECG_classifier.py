"""
run_ECG_classifier.py
---------------------
This module provides function for preparing the final predictions for evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import pickle

# Local imports
from kardioml import WORKING_PATH


def run_12ECG_classifier(data, header_data, classes, model):
    """Get predictions."""
    current_label, current_score = model.challenge_prediction(data=data, header_data=header_data)

    return current_label, current_score


def load_12ECG_model():
    """Load Physionet2017 Model"""
    # Unpickle data model
    with open(
        os.path.join(WORKING_PATH, 'models', 'physionet2017', 'physionet2017.model'), "rb"
    ) as input_file:
        phyionet2017_model = pickle.load(input_file)

    return phyionet2017_model
