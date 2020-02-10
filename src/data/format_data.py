"""
format_data.py
--------------
This module provides classes and methods for formatting the Physionet2020 dataset.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import json
import shutil
import numpy as np
import scipy.io as sio

# Local imports
from src import DATA_PATH, DATA_FILE_NAME, EXTRACTED_FOLDER_NAME, AMP_CONVERSION


class FormatData(object):

    """
    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
    https://physionetchallenges.github.io/2020/
    """

    def __init__(self):

        # Set attributes
        self.raw_path = os.path.join(DATA_PATH, 'raw')
        self.formatted_path = os.path.join(DATA_PATH, 'formatted')

    def format(self, extract=True):
        """Format Physionet2020 dataset."""
        print('Formatting Physionet2020 dataset...')
        # Extract data file
        if extract:
            self._extract_data()

        # Format data
        self._format_data()

    def _format_data(self):
        """Format raw data to standard structure."""
        # Create directory for formatted data
        os.makedirs(self.formatted_path, exist_ok=True)

        # Get a list of filenames
        filenames = [filename.split('.')[0] for filename in os.listdir(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAME))
                     if 'mat' in filename]

        for filename in filenames:
            self._format_sample(filename=filename)

    def _format_sample(self, filename):
        """Format individual .mat and .hea sample."""
        # Import matlab file
        data = self._load_mat_file(filename=filename) / AMP_CONVERSION

        # Import header file
        channel_order, age, sex, labels = self._load_header_file(filename=filename)

        # Save waveform data npy file
        np.save(os.path.join(self.formatted_path, '{}.npy'.format(filename)), data)

        # Save meta data JSON
        with open(os.path.join(self.formatted_path, '{}.json'.format(filename)), 'w') as file:
            json.dump({'filename': filename, 'channel_order': channel_order, 'age': age, 'sex': sex,
                       'labels': labels, 'shape': data.shape}, file, sort_keys=True)

    def _extract_data(self):
        """Extract the raw dataset file."""
        print('Extracting dataset...')
        shutil.unpack_archive(os.path.join(self.raw_path, DATA_FILE_NAME), self.raw_path)

    def _load_mat_file(self, filename):
        """Load Matlab waveform file."""
        return sio.loadmat(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAME, '{}.mat'.format(filename)))['val']

    def _load_header_file(self, filename):
        """Load header file."""
        # Load file
        file = open(os.path.join(self.raw_path, EXTRACTED_FOLDER_NAME, '{}.hea'.format(filename)), 'r')
        content = file.read().split('\n')
        file.close()

        # Get patient attributes
        channel_order = [row.split(' ')[-1].strip() for row in content[1:13]]
        age = content[13].split(':')[-1].strip()
        sex = content[14].split(':')[-1].strip()
        labels = [label for label in content[15].split(':')[-1].strip().split(',')]

        return channel_order, age, sex, labels
