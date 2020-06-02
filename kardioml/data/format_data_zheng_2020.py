"""
format_data_zheng_2020.py
-------------------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import copy
import json
import shutil
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer

# Local imports
from kardioml import DATA_PATH, ECG_LEADS, LABELS_LOOKUP, LABELS_COUNT, SNOMEDCT_LOOKUP, FS


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
            for index in diagnostics.index:
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

        try:
            # Import csv file
            waveforms = self._load_csv_file(filename=filename)

            # Get rpeaks
            rpeaks = self._get_rpeaks(waveforms=waveforms)

            # Normalize waveforms
            waveforms = self._scale_waveforms(waveforms=waveforms, rpeaks=rpeaks)

            # Save waveform data npy file
            np.save(os.path.join(self.formatted_path, '{}.npy'.format(filename)), waveforms)

            if labels:
                # Save meta data JSON
                with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
                    json.dump({'filename': filename,
                               'channel_order': ECG_LEADS,
                               'age': int(age),
                               'sex': sex.capitalize(),
                               'labels': labels,
                               'labels_full': [LABELS_LOOKUP[label]['label_full'] for label in labels],
                               'labels_int': [LABELS_LOOKUP[label]['label_int'] for label in labels],
                               'labels_SNOMEDCT': [SNOMEDCT_LOOKUP[label] for label in labels],
                               'label_train': self._get_training_label(labels=labels),
                               'shape': waveforms.shape,
                               'hr': int(hr),
                               'rpeaks': rpeaks},
                              file, sort_keys=True)

            else:
                # Save meta data JSON
                with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
                    json.dump({'filename': filename,
                               'channel_order': ECG_LEADS,
                               'age': int(age),
                               'sex': sex.capitalize(),
                               'labels': None,
                               'labels_full': None,
                               'labels_int': None,
                               'labels_SNOMEDCT': None,
                               'label_train': None,
                               'shape': waveforms.shape,
                               'hr': int(hr),
                               'rpeaks': rpeaks},
                              file, sort_keys=True)

        except Exception:
            pass

    def _extract_data(self):
        """Extract the raw dataset file."""
        print('Extracting dataset...')
        shutil.unpack_archive(os.path.join(self.raw_path, 'ECGDataDenoised.zip'), self.raw_path)

    def _load_csv_file(self, filename):
        """Load Matlab waveform file."""
        return np.loadtxt(open(os.path.join(self.raw_path, 'ECGDataDenoised', '{}.csv'.format(filename))),
                          delimiter=',')

    @staticmethod
    def _get_training_label(labels):
        """Return one-hot training label."""
        # Initialize binarizer
        mlb = MultiLabelBinarizer(classes=np.arange(LABELS_COUNT).tolist())

        return mlb.fit_transform([[LABELS_LOOKUP[label]['label_int'] for label in labels]])[0].tolist()

    @staticmethod
    def _get_rpeaks(waveforms):
        """Calculate median heart rate."""
        rpeaks = list()
        length = waveforms.shape[0]
        waveforms = np.pad(waveforms, ((200, 200), (0, 0)), 'constant', constant_values=0)
        for channel in range(waveforms.shape[1]):
            try:
                ecg_object = ecg.ecg(signal=waveforms[:, channel], sampling_rate=FS, show=False)
                peaks = ecg_object['rpeaks'] - 200
                peak_ids = np.where((peaks > 2) & (peaks < length - 2))[0]
                rpeaks.append(peaks[peak_ids].tolist())
            except Exception:
                pass

        return rpeaks if len([rpeak for rpeak in rpeaks if len(rpeaks) > 0]) > 0 else None

    @staticmethod
    def _scale_waveforms(waveforms, rpeaks):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if rpeaks:
            for rpeak_array in rpeaks:
                if rpeak_array:
                    return waveforms / np.median(waveforms[rpeaks[0], 0])
        return (waveforms - waveforms.mean()) / waveforms.std()
