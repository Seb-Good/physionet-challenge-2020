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
from tqdm import tqdm
import scipy.io as sio
from biosppy.signals import ecg
from joblib import Parallel, delayed
from scipy.signal.windows import blackmanharris

# Local imports
from kardioml import DATA_PATH, DATA_FILE_NAMES, EXTRACTED_FOLDER_NAMES
from kardioml.data.data_loader import parse_header
from kardioml.data.p_t_wave_detector import PTWaveDetection


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

    def format(self, extract=True, debug=False, p_and_t_waves=False, parallel=False):
        """Format Physionet2020 dataset."""
        print('Formatting Physionet2020 dataset...')
        # Extract data file
        if extract:
            self._extract_data()

        # Format data
        self._format_data(debug=debug, p_and_t_waves=p_and_t_waves, parallel=parallel)

    def _format_data(self, debug, p_and_t_waves, parallel):
        """Format raw data to standard structure."""
        # Create directory for formatted data
        os.makedirs(self.formatted_path, exist_ok=True)

        # Get a list of filenames
        filenames = [filename.split('.')[0] for filename in
                     os.listdir(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAMES[self.dataset]))
                     if 'mat' in filename]

        if debug:
            for filename in filenames[0:10]:
                self._format_sample(filename=filename, p_and_t_waves=p_and_t_waves)
        elif parallel:
            _ = Parallel(n_jobs=-1)(delayed(self._format_sample)(filename, p_and_t_waves) for filename in filenames)
        else:
            for idx in tqdm(range(len(filenames))):
                self._format_sample(filename=filenames[idx], p_and_t_waves=p_and_t_waves)

    def _format_sample(self, filename, p_and_t_waves):
        """Format individual .mat and .hea sample."""
        # Import header file
        header = self._load_header_file(filename=filename)

        # Get labels
        labels_scored = self._get_scored_labels(labels=header['labels_SNOMEDCT'])
        labels_unscored = self._get_unscored_labels(labels=header['labels_SNOMEDCT'])

        # Import matlab file
        waveforms = self._load_mat_file(filename=filename)

        # Get rpeaks
        rpeaks = self._get_rpeaks(waveforms=waveforms, fs=header['fs'])
        rpeak_array = self._get_peak_array(waveforms=waveforms, peaks=rpeaks)
        rpeak_times = self._get_peak_times(waveforms=waveforms, peak_array=rpeak_array, fs=header['fs'])

        # Get P-waves and T-waves
        if p_and_t_waves:
            p_waves, t_waves = self._get_p_and_t_waves(waveforms=waveforms, rpeaks=rpeaks)
        else:
            p_waves = [None for _ in range(header['num_leads'])]
            t_waves = [None for _ in range(header['num_leads'])]
        p_wave_array = self._get_peak_array(waveforms=waveforms, peaks=p_waves)
        p_wave_times = self._get_peak_times(waveforms=waveforms, peak_array=p_wave_array, fs=header['fs'])
        t_wave_array = self._get_peak_array(waveforms=waveforms, peaks=t_waves)
        t_wave_times = self._get_peak_times(waveforms=waveforms, peak_array=t_wave_array, fs=header['fs'])

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
                       'p_waves': p_waves,
                       'p_wave_array': p_wave_array.tolist(),
                       'p_wave_times': p_wave_times,
                       't_waves': t_waves,
                       't_wave_array': t_wave_array.tolist(),
                       't_wave_times': t_wave_times,
                       'labels_unscored_SNOMEDCT': [label['SNOMED CT Code'] for label in
                                                    labels_unscored] if labels_unscored else None,
                       'labels_unscored': [label['Abbreviation'] for label in
                                           labels_unscored] if labels_unscored else None,
                       'labels_unscored_full': [label['Dx'] for label in
                                                labels_unscored] if labels_unscored else None,
                       'p_and_t_waves': p_and_t_waves
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
        """Find rpeaks."""
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

    @staticmethod
    def _get_peak_array(waveforms, peaks):
        """Return a binary array of contiguous peak sections."""
        # Create empty array with length of waveform
        peak_array = np.zeros(waveforms.shape[0], dtype=np.float32)
        window = blackmanharris(21)
        if len([True for peak_ids in peaks if peak_ids is not None]) >= 1:
            for peak_ids in peaks:
                if peak_ids:
                    for peak_id in peak_ids:
                        if len(peak_array[peak_id - 10:peak_id + 11]) >= 21:
                            peak_array[peak_id-10:peak_id+11] += window
            peak_array[peak_array <= 1] = 0
            peak_array /= np.max(peak_array)
        return peak_array

    def _get_peak_times(self, waveforms, peak_array, fs):
        """Get list of start and end times for peaks."""
        # Get contiguous sections
        sections = self._contiguous_regions(peak_array == 1).tolist()

        # Get time array
        time = np.arange(waveforms.shape[0]) * 1 / fs

        return [[time[section[0]], time[section[1]-1]] for section in sections]

    @staticmethod
    def _get_p_and_t_waves(waveforms, rpeaks):
        """Calculate median heart rate."""
        p_waves = list()
        t_waves = list()
        for channel in range(waveforms.shape[1]):
            try:
                waves = PTWaveDetection().run(waveforms[:, channel], rpeaks[channel])
                p_waves.append(waves[0])
                t_waves.append(waves[1])
            except Exception:
                p_waves.append([])
                t_waves.append([])

        return (p_waves if len([p_wave for p_wave in p_waves if len(p_waves) > 0]) > 0 else None,
                t_waves if len([t_wave for t_wave in t_waves if len(t_waves) > 0]) > 0 else None)

    @staticmethod
    def _other_p_and_t_wave(waveforms, rpeaks, fs):
        waveform_stacked = np.sum(waveforms, axis=1) / waveforms.shape[1]
        ecg_object = ecg.ecg(signal=waveform_stacked, sampling_rate=fs, show=False)
        rpeaks = ecg_object['rpeaks']
        waves = PTWaveDetection().run(waveform_stacked, rpeaks)
        rpeak_array = np.zeros(waveforms.shape[0], dtype=np.float32)
        p_wave_array = np.zeros(waveforms.shape[0], dtype=np.float32)
        t_wave_array = np.zeros(waveforms.shape[0], dtype=np.float32)
        rpeak_array[rpeaks] = 1
        p_wave_array[waves[0]] = 1
        t_wave_array[waves[1]] = 1
        return rpeak_array, p_wave_array, t_wave_array

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
