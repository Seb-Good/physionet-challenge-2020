"""
run_ECG_classifier.py
---------------------
This module provides function for preparing the final predictions for evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import pickle


def run_12ECG_classifier(data, header_data, model):
    """Get predictions.
    Input
    -----
    data: .mat file
    header_data: .hea file
    model: pytorch model file/custom model class.

    Output
    ------
    current_label: [0, 1, 0, 0, ... , 0, 0]
    current_score: [0.1, 0.92, 0.2, 0.23, ... , 0.01, 0.002]
    classes: ['270492004', '164889003', ... , '17338001']
    """
    current_label, current_score, classes = model.predict(data=data, header_data=header_data)

    return current_label, current_score, classes


def load_12ECG_model(model_input):
    """Load Physionet2017 Model
    model_input: This is an argument from running driver.py on command line. I think we just ignore it and hard-code
    out model path.
    """
    dmitrii_model = pytorch.load_model('where_did_you_save_the_model/model_12.ckp')

    return dmitrii_model
