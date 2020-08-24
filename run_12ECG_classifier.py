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
import numpy as np


def run_12ECG_classifier(data, header_data, models):
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
                                                 fs_resampled=1000, p_and_t_waves=True)



    # Get soft predictions
    postprocessing = PostProcessing(0)
    scores = []
    current_score = np.zeros((27))
    for model in models:
        soft_pred = model.inference(X=waveforms, y=meta_data)
        current_score += soft_pred.reshape(27) / len(models)
        # get hard predictions
        scores.append(postprocessing.run(soft_pred).reshape(27))

    scores = np.array(scores)

    # majority voting
    scores = np.sum(scores, axis=0)
    scores[scores <= 3] = 0
    scores[scores > 3] = 1
    scores = scores.astype(np.int64)

    if len(np.where(scores > 0)[0]) < 1:
        scores[:] = 1

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

    current_label = scores.tolist()
    current_score = current_score.tolist()

    return current_label, current_score, classes


def load_12ECG_model(model_input):
    """Load Physionet2017 Model
    model_input: This is an argument from running driver.py on command line. I think we just ignore it and hard-code
    out model path.
    """

    models_list = [
        'ecgnet_0_fold_0.631593191670484',
        'ecgnet_1_fold_0.6370736239012214',
        'ecgnet_2_fold_0.6444454717434089',
        'ecgnet_3_fold_0.6195938932528102',
        'ecgnet_4_fold_0.6149398148500164',
        'ecgnet_5_fold_0.6409127451470004'
    ]

    os.makedirs(model_input+'/pretrained/', exist_ok=True)

    # load the model
    models = []
    for i in models_list:
        model_stack = Model(input_size=19000, n_channels=15, hparams=hparams, gpu=[], inference=True)
        model_stack.model_load("./inference_models/"+i+".pt")
        model_stack.model_save(model_input+'/pretrained/'+i+".pt")
        models.append(model_stack)

    return models



