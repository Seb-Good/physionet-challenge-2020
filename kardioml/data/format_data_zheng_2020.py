"""
format_data_zheng_2020.py
-------------------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import json
import shutil
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer

# Local imports
from kardioml import DATA_PATH,  ECG_LEADS, LABELS_LOOKUP, LABELS_COUNT
from kardioml import DATA_PATH, ECG_LEADS, LABELS_LOOKUP, LABELS_COUNT


LABEL_MAPPINGS = {'1AVB': 'I-AVB',
                  'APB': 'PAC',
                  'VPB': 'PVC',
                  'STDD': 'STD',
                  'STTU': 'STE',
                  'LBBB': 'LBBB',
                  'RBBB': 'RBBB',
                  'SR': 'Normal',
                  'AFIB': 'AF'}


class FormatDataZheng2020(object):

    """
    Classification of 12-lead ECGs: Zheng et al. (2020)
    https://figshare.com/collections/ChapmanECG/4560497/2
    """

    def __init__(self):

        # Set attributes
        self.raw_path = os.path.join(DATA_PATH, 'zheng_2020', 'raw')
        self.formatted_path = os.path.join(DATA_PATH, 'zheng_2020', 'formatted')

    def format(self, extract=True, debug=False):
        """Format Physionet2020 dataset."""
        print('Formatting Zheng et al. (2020) dataset...')
        # Extract data file
        if extract:
            self._extract_data()

        # Format data
        self._format_data(debug=debug)

    def _format_data(self, debug):
        """Format raw data to standard structure."""
        # Create directory for formatted data
        os.makedirs(self.formatted_path, exist_ok=True)

        # Load diagnostics
        diagnostics = pd.read_excel(os.path.join(self.raw_path, 'Diagnostics.xlsx'))

        if debug:
            for index in diagnostics.index[0:10]:
                self._format_sample(index=index, diagnostics=diagnostics)

        else:
            _ = Parallel(n_jobs=-1)(delayed(self._format_sample)(index, diagnostics) for index in diagnostics.index)

    def _format_sample(self, index, diagnostics):
        """Format individual .mat and .hea sample."""
        # Get attributes
        filename = diagnostics.loc[index, 'FileName']
        rhythm = diagnostics.loc[index, 'Rhythm']
        beats = diagnostics.loc[index, 'Beat'].split(' ')
        age = diagnostics.loc[index, 'PatientAge']
        sex = diagnostics.loc[index, 'Gender']
        hr = diagnostics.loc[index, 'VentricularRate']

        # Get relevant labels
        beats = [LABEL_MAPPINGS[beat] for beat in beats if beat in LABEL_MAPPINGS.keys()]
        if rhythm in LABEL_MAPPINGS.keys():
            rhythm = [LABEL_MAPPINGS[rhythm]]
        else:
            rhythm = list()
        labels = beats + rhythm

        if labels:
            # Import csv file
            data = self._load_csv_file(filename=filename)

            # Save waveform data npy file
            np.save(os.path.join(self.formatted_path, '{}.npy'.format(filename)), data)

            # Save meta data JSON
            with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
                json.dump({'filename': filename, 'channel_order': ECG_LEADS, 'age': int(age), 'sex': sex.capitalize(),
                           'labels': labels, 'labels_full': [LABELS_LOOKUP[label]['label_full'] for label in labels],
                           'labels_int': [LABELS_LOOKUP[label]['label_int'] for label in labels],
                           'label_train': self._get_training_label(labels=labels), 'shape': data.shape,
                           'hr': int(hr)},
                          file, sort_keys=True)

        else:
            pass

    def _extract_data(self):
        """Extract the raw dataset file."""
        print('Extracting dataset...')
        shutil.unpack_archive(os.path.join(self.raw_path, 'ECGDataDenoised.zip'), self.raw_path)

    def _load_csv_file(self, filename):
        """Load Matlab waveform file."""
        return np.loadtxt(open(os.path.join(self.raw_path, 'ECGDataDenoised', '{}.csv'.format(filename))),
                          delimiter=',').T

    @staticmethod
    def _get_training_label(labels):
        """Return one-hot training label."""
        # Initialize binarizer
        mlb = MultiLabelBinarizer(classes=np.arange(LABELS_COUNT).tolist())

        return mlb.fit_transform([[LABELS_LOOKUP[label]['label_int'] for label in labels]])[0].tolist()


