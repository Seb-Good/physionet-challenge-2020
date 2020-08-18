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
from predict import Predict
from config import Model,hparams
from postprocessing import PostProcessing

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

    postprocessing = PostProcessing(0)

    # Run ETL process
    waveforms, meta_data = inference_data_loader(waveforms=data, header=header_data,
                                                 fs_resampled=1000, p_and_t_waves=True)

    # Get soft predictions
    current_score = model.inference(X=waveforms, y=meta_data)

    # get hard predictions
    current_label = postprocessing.run(current_score)

    classes = ['270492004',
            '164889003',
            '164890007',
            '426627000',
            '713427006',
            '713426002',
            '445118002',
            '39732003',
            '164909002',
            '251146004',
            '698252002',
            '10370003',
            '284470004',
            '427172004',
            '164947007',
            '111975006',
            '164917005',
            '47665007',
            '59118001',
            '427393009',
            '426177001',
            '426783006',
            '427084000',
            '63593006',
            '164934002',
            '59931005',
            '17338001',
               ]

    return current_label[0,:].tolist(), current_score[0,:].tolist(), classes


def load_12ECG_model(model_input):
    """Load Physionet2017 Model
    model_input: This is an argument from running driver.py on command line. I think we just ignore it and hard-code
    out model path.
    """


    # load the model
    model = Model(input_size=19000, n_channels=12, hparams=hparams, gpu=[])
    model.model_load("./inference_models/ecgnet_0_fold_0.6078759902401878.pt")



    return model
