#!/usr/bin/env python

# Third pary imports
import copy
import numpy as np, os
import pandas as pd
import json
from joblib import Parallel, delayed
from scipy import signal
from biosppy.signals import ecg
from scipy.signal.windows import blackmanharris
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.io as sio

# Local imports
from kardioml import DATA_PATH
from kardioml.data.data_loader import parse_header
from kardioml.data.p_t_wave_detector import PTWaveDetection
import os
from config import hparams,Model,SPLIT_TABLE_PATH,SPLIT_TABLE_NAME,DEBUG_FOLDER
from cv_pipeline import CVPipeline

def train_12ECG_classifier(input_directory, output_directory):

    """Process Data"""
    print('Loading data...')

    # Create directory for processed training data
    input_directory_processed = os.path.join(DATA_PATH, 'input_directory_processed')
    os.makedirs(input_directory_processed, exist_ok=True)

    # Get a list of filenames
    filenames = [filename.split('.')[0] for filename in os.listdir(input_directory) if 'mat' in filename]

    # Set sample rate for processed waveforms
    fs_resampled = 1000

    # Do you want to extract PT-waves?
    p_and_t_waves = True

    _ = Parallel(n_jobs=-1)(delayed(data_loader)(filename, input_directory, input_directory_processed,
                                                 fs_resampled, p_and_t_waves) for filename in filenames)


    os.makedirs(output_directory, exist_ok=True)


    """Process Data"""
    print('Training model...')
    # Train the classifier
    for start_fold in range(6):

        gpu = []

        hparams['lr'] = 0.001
        hparams['batch_size'] = 64
        hparams['start_fold'] = int(start_fold)
        hparams['n_epochs'] = 100

        hparams['model_path'] = './'+output_directory
        hparams['checkpoint_path'] = hparams['model_path'] + '/checkpoint'
        hparams['model_name'] = '/ecgnet'

        hparams[''] = output_directory

        try:
            cross_val = CVPipeline(
                hparams=hparams,
                split_table_path=SPLIT_TABLE_PATH,
                split_table_name=SPLIT_TABLE_NAME,
                debug_folder=DEBUG_FOLDER,
                model=Model,
                gpu=gpu,
                downsample=False
            )
        except:
            pass



def data_loader(filename, input_directory, input_directory_processed, fs_resampled, p_and_t_waves=False):
    """Convert data and header_data to .npy and dict format."""
    # Dataset lookup
    lookup = {'A': 'A', 'Q': 'B', 'I': 'C', 'S': 'D', 'H': 'E', 'E': 'F'}

    # Get datset
    dataset = lookup[filename[0]]

    # Import header file
    header = _load_header_file(filename=filename, input_directory=input_directory)

    # Get labels
    labels = Labels(labels_SNOMEDCT=header['labels_SNOMEDCT'])

    # Import matlab file
    waveforms = _load_mat_file(filename=filename, input_directory=input_directory)

    # Resample waveforms
    samples = int(waveforms.shape[0] * fs_resampled / header['fs'])
    waveforms = signal.resample(x=waveforms, num=samples, axis=0)

    # Compute heart rate
    hr = _compute_heart_rate(waveforms=waveforms, fs=fs_resampled)

    # Get rpeaks
    rpeaks = _get_rpeaks(waveforms=waveforms, fs=fs_resampled)
    rpeak_array = _get_peak_array(waveforms=waveforms, peaks=rpeaks)
    rpeak_times = _get_peak_times(waveforms=waveforms, peak_array=rpeak_array, fs=fs_resampled)

    # Get P-waves and T-waves
    if p_and_t_waves:
        p_waves, t_waves = _get_p_and_t_waves(waveforms=waveforms, rpeaks=rpeaks)
    else:
        p_waves = None
        t_waves = None
    p_wave_array = _get_peak_array(waveforms=waveforms, peaks=p_waves)
    p_wave_times = _get_peak_times(waveforms=waveforms, peak_array=p_wave_array, fs=fs_resampled)
    t_wave_array = _get_peak_array(waveforms=waveforms, peaks=t_waves)
    t_wave_times = _get_peak_times(waveforms=waveforms, peak_array=t_wave_array, fs=fs_resampled)

    os.makedirs(os.path.join(input_directory_processed, dataset, 'formatted'), exist_ok=True)

    # Save waveform data npy file
    np.save(os.path.join(input_directory_processed, dataset, 'formatted', '{}.npy'.format(filename)), waveforms)

    # Save meta data JSON
    with open(os.path.join(input_directory_processed, dataset, 'formatted', '{}.json'.format(filename)), 'w') as file:
        json.dump({'filename': filename,
                   'dataset': dataset,
                   'datetime': header['datetime'],
                   'channel_order': header['channel_order'],
                   'age': header['age'],
                   'sex': header['sex'],
                   'amp_conversion': header['amp_conversion'],
                   'fs': header['fs'],
                   'fs_resampled': fs_resampled,
                   'length': header['length'],
                   'num_leads': header['num_leads'],
                   'labels_SNOMEDCT': labels.labels_SNOMEDCT,
                   'labels_short': labels.labels_short,
                   'labels_full': labels.labels_full,
                   'labels_int': labels.labels_int,
                   'labels_training': labels.labels_training,
                   'labels_training_merged': labels.labels_training_merged,
                   'shape': waveforms.shape,
                   'hr': hr,
                   'rpeaks': rpeaks,
                   'rpeak_array': rpeak_array.tolist(),
                   'rpeak_times': rpeak_times,
                   'p_waves': p_waves,
                   'p_wave_array': p_wave_array.tolist(),
                   'p_wave_times': p_wave_times,
                   't_waves': t_waves,
                   't_wave_array': t_wave_array.tolist(),
                   't_wave_times': t_wave_times,
                   'labels_unscored_SNOMEDCT': labels.labels_unscored_SNOMEDCT,
                   'labels_unscored_short': labels.labels_unscored_short,
                   'labels_unscored_full': labels.labels_unscored_full,
                   'p_and_t_waves': p_and_t_waves
                   },
                  file, sort_keys=False, indent=4)


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


def _get_peak_array(waveforms, peaks):
    """Return a binary array of contiguous peak sections."""
    # Create empty array with length of waveform
    peak_array = np.zeros(waveforms.shape[0], dtype=np.float32)
    window = blackmanharris(21)
    if peaks:
        for peak_ids in peaks:
            if peak_ids:
                for peak_id in peak_ids:
                    if len(peak_array[peak_id - 10:peak_id + 11]) >= 21:
                        peak_array[peak_id - 10:peak_id + 11] += window
        peak_array[peak_array <= 1] = 0
        peak_array /= np.max(peak_array)
    return peak_array


def _get_peak_times(waveforms, peak_array, fs):
    """Get list of start and end times for peaks."""
    # Get contiguous sections
    sections = _contiguous_regions(peak_array >= 0.5).tolist()

    # Get time array
    time = np.arange(waveforms.shape[0]) * 1 / fs

    return [[time[section[0]], time[section[1] - 1]] for section in sections]


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


def _load_header_file(filename, input_directory):
    """Load header file."""
    # Load file
    file = open(os.path.join(input_directory, '{}.hea'.format(filename)), 'r')
    content = file.read().split('\n')
    file.close()
    return parse_header(header_data=content)


def _load_mat_file(filename, input_directory):
    """Load Matlab waveform file."""
    return sio.loadmat(os.path.join(input_directory, '{}.mat'.format(filename)))['val'].T


class Labels(object):

    def __init__(self, labels_SNOMEDCT):

        # Set parameters
        self.labels_SNOMEDCT = labels_SNOMEDCT

        # Scored labels
        self.labels_scored_lookup = pd.read_csv(os.path.join(DATA_PATH, 'labels_scored.csv'))
        self.labels_scored = self._get_scored_labels(labels=self.labels_SNOMEDCT)
        self.labels_short = [label['Abbreviation'] for label in self.labels_scored] if self.labels_scored else None
        self.labels_full = [label['Dx'] for label in self.labels_scored] if self.labels_scored else None
        self.labels_int = [int(self.labels_scored_lookup[self.labels_scored_lookup['SNOMED CT Code'] ==
                                                         label['SNOMED CT Code']].index[0])
                           for label in self.labels_scored] if self.labels_scored else None
        self.labels_training = self._get_training_label(labels=self.labels_int) if self.labels_int else None
        self.labels_training_merged = self._get_merged_label()

        # Unscored labels
        self.labels_unscored_lookup = pd.read_csv(os.path.join(DATA_PATH, 'labels_unscored.csv'))
        self.labels_unscored = self._get_unscored_labels(labels=self.labels_SNOMEDCT)
        self.labels_unscored_SNOMEDCT = [label['SNOMED CT Code'] for label in self.labels_unscored] if self.labels_unscored else None
        self.labels_unscored_short = [label['Abbreviation'] for label in self.labels_unscored] if self.labels_unscored else None
        self.labels_unscored_full = [label['Dx'] for label in self.labels_unscored] if self.labels_unscored else None

    def _get_scored_labels(self, labels):
        """Return a list scored labels."""
        labels_list = list()
        for label in labels:
            row = self.labels_scored_lookup[self.labels_scored_lookup['SNOMED CT Code'] == label]
            if row.shape[0] > 0:
                labels_list.append(row.to_dict(orient='row')[0])
        if len(labels_list) > 0:
            return labels_list
        return None

    def _get_unscored_labels(self, labels):
        """Return a list scored labels."""
        labels_list = list()
        for label in labels:
            row = self.labels_unscored_lookup[self.labels_unscored_lookup['SNOMED CT Code'] == label]
            if row.shape[0] > 0:
                labels_list.append(row.to_dict(orient='row')[0])
        if len(labels_list) > 0:
            return labels_list
        return None

    def _get_training_label(self, labels):
        """Return one-hot training label."""
        mlb = MultiLabelBinarizer(classes=self.labels_scored_lookup.index.tolist())
        return mlb.fit_transform([labels])[0].tolist()

    def _get_merged_label(self):
        label_maps = [[4, 18], [12, 23], [13, 26]]
        if self.labels_training:
            labels_training = copy.copy(self.labels_training)
            for label in label_maps:
                if labels_training[label[1]] == 1:
                    labels_training[label[1]] = 0
                    labels_training[label[0]] = 1
            return labels_training
        return None



train_12ECG_classifier('input')