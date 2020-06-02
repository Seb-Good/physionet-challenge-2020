"""
format_data_physionet_2020.py
-----------------------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import copy
import json
import shutil
import numpy as np
import scipy.io as sio
from biosppy.signals import ecg
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer

# Local imports
from kardioml import (DATA_PATH, DATA_FILE_NAMES, EXTRACTED_FOLDER_NAMES, AMP_CONVERSION,
                      LABELS_LOOKUP, LABELS_COUNT, FS, SNOMEDCT_LOOKUP)


class FormatDataPhysionet2020(object):

    """
    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
    https://physionetchallenges.github.io/2020/
    """

    def __init__(self, tranche):

        # Set parameters
        self.tranche = tranche

        # Set attributes
        self.raw_path = os.path.join(DATA_PATH, 'physionet_2020_{}'.format(self.tranche), 'raw')
        self.formatted_path = os.path.join(DATA_PATH, 'physionet_2020_{}'.format(self.tranche), 'formatted')

    def format(self, extract=True, debug=False):
        """Format Physionet2020 dataset."""
        print('Formatting Physionet2020 dataset...')
        # Extract data file
        if extract:
            self._extract_data()

        # Format data
        self._format_data(debug=debug)

    def _format_data(self, debug):
        """Format raw data to standard structure."""
        # Create directory for formatted data
        os.makedirs(self.formatted_path, exist_ok=True)

        # Get a list of filenames
        filenames = [filename.split('.')[0] for filename in
                     os.listdir(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.tranche-1]))
                     if 'mat' in filename]

        if debug:
            for filename in filenames[0:10]:
                self._format_sample(filename=filename)

        else:
            _ = Parallel(n_jobs=-1)(delayed(self._format_sample)(filename) for filename in filenames)

    def _format_sample(self, filename):
        """Format individual .mat and .hea sample."""
        # Import matlab file
        waveforms = self._load_mat_file(filename=filename) / AMP_CONVERSION

        # Get rpeaks
        rpeaks = self._get_rpeaks(waveforms=waveforms)
        rpeak_array = self._get_peak_array(waveforms=waveforms, peaks=rpeaks)
        rpeak_times = self._get_peak_times(waveforms=waveforms, peak_array=rpeak_array)

        # Import header file
        channel_order, age, sex, labels = self._load_header_file(filename=filename)

        # Get labels for evaluation
        labels = [label for label in labels if label in SNOMEDCT_LOOKUP.keys()]

        # Normalize waveforms
        waveforms = self._scale_waveforms(waveforms=waveforms, rpeaks=rpeaks)

        # Save waveform data npy file
        np.save(os.path.join(self.formatted_path, '{}.npy'.format(filename)), waveforms)

        if labels:
            # Save meta data JSON
            with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
                json.dump({'filename': filename,
                           'channel_order': channel_order,
                           'age': age,
                           'sex': sex,
                           'labels': [SNOMEDCT_LOOKUP[label] for label in labels],
                           'labels_full': [LABELS_LOOKUP[SNOMEDCT_LOOKUP[label]]['label_full'] for label in labels],
                           'labels_int': [LABELS_LOOKUP[SNOMEDCT_LOOKUP[label]]['label_int'] for label in labels],
                           'labels_SNOMEDCT': labels,
                           'label_train': self._get_training_label(labels=[SNOMEDCT_LOOKUP[label] for label in labels]),
                           'shape': waveforms.shape,
                           'hr': self._compute_heart_rate(waveforms=waveforms),
                           'rpeaks': rpeaks,
                           'rpeak_array': rpeak_array.tolist(),
                           'rpeak_times': rpeak_times},
                          file, sort_keys=True)

        else:
            # Save meta data JSON
            with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
                json.dump({'filename': filename,
                           'channel_order': channel_order,
                           'age': age,
                           'sex': sex,
                           'labels': None,
                           'labels_full': None,
                           'labels_int': None,
                           'labels_SNOMEDCT': None,
                           'label_train': None,
                           'shape': waveforms.shape,
                           'hr': self._compute_heart_rate(waveforms=waveforms),
                           'rpeaks': rpeaks,
                           'rpeak_array': rpeak_array.tolist(),
                           'rpeak_times': rpeak_times},
                          file, sort_keys=True)

    @staticmethod
    def _compute_heart_rate(waveforms):
        """Calculate median heart rate."""
        hr = list()
        for channel in range(waveforms.shape[1]):
            try:
                ecg_object = ecg.ecg(signal=waveforms[:, channel], sampling_rate=FS, show=False)
                hr.extend(ecg_object['heart_rate'])
            except Exception:
                pass

        return np.median(hr) if len(hr) > 0 else 'nan'

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

    def _get_peak_array(self, waveforms, peaks):
        """Return a binary array of contiguous peak sections."""
        # Create empty array with length of waveform
        peak_array = np.zeros(waveforms.shape[0], dtype=np.float32)

        if peaks:
            for peak_ids in peaks:
                if peak_ids:
                    peak_array[peak_ids] = 1

        sections = self._contiguous_regions(peak_array == 0)
        for section in sections.tolist():
            if section[1] - section[0] <= 30:
                peak_array[section[0]:section[1]] = 1

        sections = self._contiguous_regions(peak_array == 1)
        for section in sections.tolist():
            if section[1] - section[0] == 1:
                peak_array[section[0]:section[1]] = 0

        return peak_array

    def _get_peak_times(self, waveforms, peak_array):
        """Get list of start and end times for peaks."""
        # Get contiguous sections
        sections = self._contiguous_regions(peak_array == 1).tolist()

        # Get time array
        time = np.arange(waveforms.shape[0]) * 1 / FS

        return [[time[section[0]], time[section[1]-1]] for section in sections]

    def _extract_data(self):
        """Extract the raw dataset file."""
        print('Extracting dataset...')
        shutil.unpack_archive(os.path.join(self.raw_path, DATA_FILE_NAMES[self.tranche-1]), self.raw_path)

    def _load_mat_file(self, filename):
        """Load Matlab waveform file."""
        return sio.loadmat(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.tranche-1],
                                        '{}.mat'.format(filename)))['val'].T

    def _load_header_file(self, filename):
        """Load header file."""
        # Load file
        file = open(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.tranche-1], '{}.hea'.format(filename)), 'r')
        content = file.read().split('\n')
        file.close()

        # Get patient attributes
        channel_order = [row.split(' ')[-1].strip() for row in content[1:13]]
        age = content[13].split(':')[-1].strip()
        sex = content[14].split(':')[-1].strip()
        labels = [label for label in content[15].split(':')[-1].strip().split(',')]

        return channel_order, age, sex, labels

    @staticmethod
    def _get_training_label(labels):
        """Return one-hot training label."""
        # Initialize binarizer
        mlb = MultiLabelBinarizer(classes=np.arange(LABELS_COUNT).tolist())

        return mlb.fit_transform([[LABELS_LOOKUP[label]['label_int'] for label in labels]])[0].tolist()

    @staticmethod
    def _scale_waveforms(waveforms, rpeaks):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if rpeaks:
            for rpeak_array in rpeaks:
                if rpeak_array:
                    return waveforms / np.median(waveforms[rpeaks[0], 0])
        return (waveforms - waveforms.mean()) / waveforms.std()

    @staticmethod
    def _contiguous_regions(condition):
        """Find the indices of changes in condition"""
        d = np.diff(condition)
        idx, = d.nonzero()

        # Shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]

        # Reshape the result into two columns
        idx.shape = (-1, 2)

        return idx
