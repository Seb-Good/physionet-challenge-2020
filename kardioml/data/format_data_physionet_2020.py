"""
format_data_physionet_2020.py
-----------------------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import json
import shutil
import numpy as np
import pandas as pd
import scipy.io as sio
from biosppy.signals import ecg
from joblib import Parallel, delayed

# Local imports
from kardioml import DATA_PATH, DATA_FILE_NAMES, EXTRACTED_FOLDER_NAMES
from kardioml.data.data_loader import parse_header


class FormatDataPhysionet2020(object):

    """
    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
    https://physionetchallenges.github.io/2020/
    """

    def __init__(self, dataset):

        # Set parameters
        self.dataset = dataset

        # Set attributes
        self.raw_path = os.path.join(DATA_PATH, self.dataset, 'raw')
        self.formatted_path = os.path.join(DATA_PATH, self.dataset, 'formatted')
        self.labels_scored = pd.read_csv(os.path.join(DATA_PATH, 'labels_scored.csv'))
        self.labels_unscored = pd.read_csv(os.path.join(DATA_PATH, 'labels_unscored.csv'))

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
                     os.listdir(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.dataset]))
                     if 'mat' in filename]

        if debug:
            for filename in filenames[0:10]:
                self._format_sample(filename=filename)

        else:
            _ = Parallel(n_jobs=-1)(delayed(self._format_sample)(filename) for filename in filenames)

    def _format_sample(self, filename):
        """Format individual .mat and .hea sample."""
        # Import header file
        header = self._load_header_file(filename=filename)

        # Get labels
        labels_scored = self._get_scored_labels(labels=header['labels_SNOMEDCT'])
        labels_unscored = self._get_unscored_labels(labels=header['labels_SNOMEDCT'])

        # Import matlab file
        waveforms = self._load_mat_file(filename=filename) / header['amp_conversion']

        # Get rpeaks
        rpeaks = self._get_rpeaks(waveforms=waveforms, fs=header['fs'])
        rpeak_array = self._get_peak_array(waveforms=waveforms, peaks=rpeaks)
        rpeak_times = self._get_peak_times(waveforms=waveforms, peak_array=rpeak_array, fs=header['fs'])

        # Normalize waveforms
        waveforms = self._scale_waveforms(waveforms=waveforms, rpeaks=rpeaks)

        # Save waveform data npy file
        np.save(os.path.join(self.formatted_path, '{}.npy'.format(filename)), waveforms)

        # Save meta data JSON
        with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
            json.dump({'filename': filename,
                       'datetime': header['datetime'],
                       'channel_order': header['channel_order'],
                       'age': header['age'],
                       'sex': header['sex'],
                       'amp_conversion': header['amp_conversion'],
                       'fs': header['fs'],
                       'length': header['length'],
                       'num_leads': header['num_leads'],
                       'labels_SNOMEDCT': [label['SNOMED CT Code'] for label in
                                           labels_scored] if labels_scored else None,
                       'labels': [label['Abbreviation'] for label in labels_scored] if labels_scored else None,
                       'labels_full': [label['Dx'] for label in labels_scored] if labels_scored else None,
                       'shape': waveforms.shape,
                       'hr': self._compute_heart_rate(waveforms=waveforms, fs=header['fs']),
                       'rpeaks': rpeaks,
                       'rpeak_array': rpeak_array.tolist(),
                       'rpeak_times': rpeak_times,
                       'labels_unscored_SNOMEDCT': [label['SNOMED CT Code'] for label in
                                                    labels_unscored] if labels_unscored else None,
                       'labels_unscored': [label['Abbreviation'] for label in
                                           labels_unscored] if labels_unscored else None,
                       'labels_unscored_full': [label['Dx'] for label in
                                                labels_unscored] if labels_unscored else None,
                       },
                      file, sort_keys=False, indent=4)

    def _get_scored_labels(self, labels):
        """Return a list scored labels."""
        labels_list = list()
        for label in labels:
            row = self.labels_scored[self.labels_scored['SNOMED CT Code'] == label]
            if row.shape[0] > 0:
                labels_list.append(row.to_dict(orient='row')[0])
        if len(labels_list) > 0:
            return labels_list
        return None

    def _get_unscored_labels(self, labels):
        """Return a list scored labels."""
        labels_list = list()
        for label in labels:
            row = self.labels_unscored[self.labels_unscored['SNOMED CT Code'] == label]
            if row.shape[0] > 0:
                labels_list.append(row.to_dict(orient='row')[0])
        if len(labels_list) > 0:
            return labels_list
        return None

    @staticmethod
    def _compute_heart_rate(waveforms, fs):
        """Calculate median heart rate."""
        hr = list()
        for channel in range(waveforms.shape[1]):
            try:
                ecg_object = ecg.ecg(signal=waveforms[:, channel], sampling_rate=fs, show=False)
                hr.extend(ecg_object['heart_rate'])
            except Exception:
                pass

        return np.median(hr) if len(hr) > 0 else 'nan'

    @staticmethod
    def _get_rpeaks(waveforms, fs):
        """Calculate median heart rate."""
        rpeaks = list()
        length = waveforms.shape[0]
        waveforms = np.pad(waveforms, ((200, 200), (0, 0)), 'constant', constant_values=0)
        for channel in range(waveforms.shape[1]):
            try:
                # Get + peaks
                ecg_object = ecg.ecg(signal=waveforms[:, channel], sampling_rate=fs, show=False)
                median_plus = np.median(ecg_object['filtered'][ecg_object['rpeaks']])
                peaks_plus = ecg_object['rpeaks'] - 200
                peak_ids_plus = np.where((peaks_plus > 2) & (peaks_plus < length - 2))[0]

                # Get - peaks
                ecg_object = ecg.ecg(signal=-waveforms[:, channel], sampling_rate=fs, show=False)
                median_minus = np.median(ecg_object['filtered'][ecg_object['rpeaks']])
                peaks_minus = ecg_object['rpeaks'] - 200
                peak_ids_minus = np.where((peaks_minus > 2) & (peaks_minus < length - 2))[0]

                if median_plus >= median_minus:
                    rpeaks.append(peaks_plus[peak_ids_plus].tolist())
                else:
                    rpeaks.append(peaks_minus[peak_ids_minus].tolist())
            except Exception:
                rpeaks.append([])

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

    def _get_peak_times(self, waveforms, peak_array, fs):
        """Get list of start and end times for peaks."""
        # Get contiguous sections
        sections = self._contiguous_regions(peak_array == 1).tolist()

        # Get time array
        time = np.arange(waveforms.shape[0]) * 1 / fs

        return [[time[section[0]], time[section[1]-1]] for section in sections]

    def _extract_data(self):
        """Extract the raw dataset file."""
        print('Extracting dataset...')
        shutil.unpack_archive(os.path.join(self.raw_path, DATA_FILE_NAMES[self.dataset]), self.raw_path)

    def _load_mat_file(self, filename):
        """Load Matlab waveform file."""
        return sio.loadmat(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.dataset],
                                        '{}.mat'.format(filename)))['val'].T

    def _load_header_file(self, filename):
        """Load header file."""
        # Load file
        file = open(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.dataset], '{}.hea'.format(filename)), 'r')
        content = file.read().split('\n')
        file.close()
        return parse_header(header_data=content)

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
