"""
run_ECG_classifier.py
---------------------
This module provides function for preparing the final predictions for evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import pickle


# local imports
from kardioml.data.inference_data_loader import inference_data_loader


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
    # Run ETL process
    waveforms, meta_data = inference_data_loader(waveforms=data, header=header_data,
                                                 fs_resampled=500, p_and_t_waves=True)

    # Get prediction
    current_label, current_score, classes = model.predict(waveforms=waveforms, meta_data=meta_data)

    return current_label, current_score, classes


def load_12ECG_model(model_input):
    """Load Physionet2017 Model
    model_input: This is an argument from running driver.py on command line. I think we just ignore it and hard-code
    out model path.
    """
    dmitrii_model = pytorch.load_model('where_did_you_save_the_model/model_12.ckp')

    return dmitrii_model
