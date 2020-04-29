"""
data_loader.py
--------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer

# Locals imports
from kardioml import ECG_LEADS, LABELS_LOOKUP, LABELS_COUNT


def load_challenge_data(filename):
    """
    Example of data loader from driver.py

    Input
    _____
    filename: matlab file name, example: 'A0001.mat'

    Output
    ______
    data: np.array([12, num_samples])
    array([[  28.,   39.,   45., ...,  258.,  259.,  259.],
           [   7.,   11.,   15., ...,  248.,  249.,  250.],
           [ -21.,  -28.,  -30., ...,  -10.,  -10.,   -9.],
           ...,
           [-112., -110., -108., ...,  194.,  194.,  195.],
           [-596., -590., -582., ...,  307.,  307.,  307.],
           [ -16.,   -7.,    2., ...,  213.,  214.,  214.]])

    header_data: list of text rows
    ['A0001 12 500 7500 05-Feb-2020 11:39:16\n',
     'A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I\n',
     'A0001.mat 16+24 1000/mV 16 0 7 2029 0 II\n',
     'A0001.mat 16+24 1000/mV 16 0 -21 3745 0 III\n',
     'A0001.mat 16+24 1000/mV 16 0 -17 3680 0 aVR\n',
     'A0001.mat 16+24 1000/mV 16 0 24 -2664 0 aVL\n',
     'A0001.mat 16+24 1000/mV 16 0 -7 -1499 0 aVF\n',
     'A0001.mat 16+24 1000/mV 16 0 -290 390 0 V1\n',
     'A0001.mat 16+24 1000/mV 16 0 -204 157 0 V2\n',
     'A0001.mat 16+24 1000/mV 16 0 -96 -2555 0 V3\n',
     'A0001.mat 16+24 1000/mV 16 0 -112 49 0 V4\n',
     'A0001.mat 16+24 1000/mV 16 0 -596 -321 0 V5\n',
     'A0001.mat 16+24 1000/mV 16 0 -16 -3112 0 V6\n',
     '#Age: 74\n',
     '#Sex: Male\n',
     '#Dx: RBBB\n',
     '#Rx: Unknown\n',
     '#Hx: Unknown\n',
     '#Sx: Unknows\n']
    """
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


def parse_header(header_data):
    """Parse header text."""
    filename = header_data[0].split(' ')[0]
    length = header_data[3].split(' ')[0]
    channel_order = [row.split(' ')[-1].strip() for row in header_data[1:13]]
    age = header_data[13].split(':')[-1].strip()
    sex = header_data[14].split(':')[-1].strip()
    labels = [label for label in header_data[15].split(':')[-1].strip().split(',')]

    # Get training labels
    mlb = MultiLabelBinarizer(classes=np.arange(LABELS_COUNT).tolist())
    label_train = mlb.fit_transform([[LABELS_LOOKUP[label]['label_int'] for label in labels]])[0].tolist()

    return {'filename': filename, 'channel_order': channel_order, 'age': age, 'sex': sex,
            'labels': labels, 'labels_full': [LABELS_LOOKUP[label]['label_full'] for label in labels],
            'labels_int': [LABELS_LOOKUP[label]['label_int'] for label in labels],
            'label_train': label_train, 'shape': [len(ECG_LEADS), length]}
