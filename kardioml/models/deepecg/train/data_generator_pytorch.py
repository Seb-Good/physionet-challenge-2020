
# basic libs
import numpy as np
import json
import os
import random
from scipy import signal

# pytorch
import torch
from torch.utils.data import Dataset

np.random.seed(42)



class Dataset_train(Dataset):
    def __init__(self, patients,aug):

        self.patients = patients
        self.aug = aug

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        X, y = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X, y

    def load_data(self, id, train=True):

        if self.patients[id][0] == 'A':
            data_folder = 'A'
        elif self.patients[id][0] == 'Q':
            data_folder = 'B'
        elif self.patients[id][0] == 'I':
            data_folder = 'C'
        elif self.patients[id][0] == 'S':
            data_folder = 'D'
        elif self.patients[id][0] == 'H':
            data_folder = 'E'
        elif self.patients[id][0] == 'E':
            data_folder = 'F'
        else:
            a = self.patients[id]
            print(1)

        data_folder = f'./data/{data_folder}/formatted/'

        # load waveforms
        X = np.load(data_folder+self.patients[id] + '.npy')

        # load annotation
        y = json.load(open(data_folder + self.patients[id] + '.json'))

        # Scale waveform amplitudes
        X = (X - np.mean(X)) / np.std(X)
        """
        Maybe try this (see method below).
        X = apply_amplitude_scaling(X=X, y=y)
        """

        # TODO: Seb's augmentation implementation point
        # We need a way to inform this method of the sample rate for the dataset.
        fs_training = 350
        if self.aug is True:
            X = self.apply_augmentation(waveform=X, meta_data=y, fs_training=fs_training, max_samples=19000)

        #padding
        if X.shape[0] < 38000:
            padding = np.zeros((38000 - X.shape[0], X.shape[1]))
            X = np.concatenate([X, padding], axis=0)

        return X, y['labels_training_merged']

        # if train:
        #     # load annotation
        #     y = json.load(open(data_folder+self.patients[id] + '.json'))
        #
        #     return X, y['labels_training_merged']
        # else:
        #     return X

    @staticmethod
    def apply_amplitude_scaling(X, y):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if y['rpeaks']:
            for channel_rpeaks in y['rpeaks']:
                if channel_rpeaks:
                    return X / np.median(X[y['rpeaks'][0], 0])
        return (X - X.mean()) / X.std()

    def apply_augmentation(self, waveform, meta_data, fs_training, max_samples):

        # Random resample
        waveform = self._random_resample(waveform=waveform, meta_data=meta_data,
                                         fs_training=fs_training, probability=0.25, max_samples=max_samples)

        # Random amplitude scale
        waveform = self._random_scale(waveform=waveform, probability=0.5)

        # Apply synthetic noise
        waveform = self._add_synthetic_noise(waveform=waveform, fs_training=fs_training, probability=0.25)

        return waveform

    def _random_resample(self, waveform, meta_data, fs_training, probability, max_samples):
        """Randomly resample waveform.
        bradycardia=3, sinus bradycardia=20, sinus tachycardia=22
        """
        if (
                meta_data['hr'] != 'nan' and
                all(meta_data['labels_training_merged'][label] == 0 for label in [3, 20, 22]) and
                self._coin_flip(probability=probability)
        ):
            # Get waveform duration
            duration = waveform.shape[0] / fs_training

            # Physiological limits
            hr_new = int(meta_data['hr'] * np.random.uniform(0.9, 1.1))
            if hr_new > 300:
                hr_new = 300
            elif hr_new < 40:
                hr_new = 40
            else:
                pass

            # Get new duration
            duration_new = duration * meta_data['hr'] / hr_new

            # Get number of samples

            samples = int(duration_new * fs_training)
            if samples > max_samples:
                samples = max_samples

            # Resample waveform
            waveform = signal.resample_poly(waveform, samples, waveform.shape[0], axis=0).astype(np.float32)

            return waveform
        else:
            return waveform

    def _random_scale(self, waveform, probability):
        """Apply random scale factor between 0.25 and 3 to the waveform amplitudes."""
        # Get random scale factor
        scale_factor = random.uniform(0.25, 3.)

        if self._coin_flip(probability):
            return waveform * scale_factor
        return waveform

    def _add_synthetic_noise(self, waveform, fs_training, probability):
        """Add different kinds of synthetic noise to the signal."""
        waveform = waveform.squeeze()
        for idx in range(waveform.shape[1]):
            waveform[:, idx] = self._generate_baseline_wandering_noise(waveform=waveform[:, idx],
                                                                       fs=fs_training, probability=probability)
            waveform[:, idx] = self._generate_high_frequency_noise(waveform=waveform[:, idx],
                                                                   fs=fs_training, probability=probability)
            waveform[:, idx] = self._generate_gaussian_noise(waveform=waveform[:, idx], probability=probability)
            waveform[:, idx] = self._generate_pulse_noise(waveform=waveform[:, idx], probability=probability)
        return waveform

    def _generate_baseline_wandering_noise(self, waveform, fs, probability):
        """Adds baseline wandering to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):

            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Get number of baseline signals
            baseline_signals = random.randint(1, 5)

            # Loop through baseline signals
            for baseline_signal in range(baseline_signals):
                # Add noise
                waveform += random.uniform(0.01, 0.75) * np.sin(2 * np.pi * random.uniform(0.001, 0.5) *
                                                                time + random.uniform(0, 60))

        return waveform

    def _generate_high_frequency_noise(self, waveform, fs, probability=0.5):
        """Adds high frequency sinusoidal noise to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):
            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Add noise
            waveform += random.uniform(0.001, 0.3) * np.sin(2 * np.pi * random.uniform(50, 200) *
                                                            time + random.uniform(0, 60))

        return waveform

    def _generate_gaussian_noise(self, waveform, probability=0.5):
        """Adds white noise noise to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):
            waveform += np.random.normal(loc=0.0, scale=random.uniform(0.01, 0.25), size=len(waveform))

        return waveform

    def _generate_pulse_noise(self, waveform, probability=0.5):
        """Adds gaussian pulse to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):

            # Get pulse
            pulse = signal.gaussian(int(len(waveform) * random.uniform(0.05, 0.010)), std=random.randint(50, 200))
            pulse = np.diff(pulse)

            # Get remainder
            remainder = len(waveform) - len(pulse)
            if remainder >= 0:
                left_pad = int(remainder * random.uniform(0., 1.))
                right_pad = remainder - left_pad
                pulse = np.pad(pulse, (left_pad, right_pad), 'constant', constant_values=0)
                pulse = pulse / pulse.max()

            waveform += pulse * random.uniform(waveform.max() * 1.5, waveform.max() * 2)

        return waveform

    @staticmethod
    def _coin_flip(probability):
        if random.random() < probability:
            return True
        return False

    def get_labels(self):
        """
        :param ids: a list of ids for loading from the database
        :return: y: numpy array of labels, shape(n_samples,n_labels)
        """

        for index, record in enumerate(self.patients):

            if record[0] == 'A':
                data_folder = 'A'

            elif record[0] == 'Q':
                data_folder = 'B'

            elif record[0] == 'I':
                data_folder = 'C'

            elif record[0] == 'S':
                data_folder = 'D'

            elif record[0] == 'H':
                data_folder = 'E'

            elif record[0] == 'E':
                data_folder = 'F'

            data_folder = f'./data/{data_folder}/formatted/'


            if index == 0:
                y = np.array([json.load(open(data_folder+record + '.json'))['labels_training_merged']])
                y = np.reshape(y, (1, 27))
            else:
                temp = np.array([json.load(open(data_folder+record + '.json'))['labels_training_merged']])
                temp = np.reshape(temp, (1, 27))
                y = np.concatenate((y, temp), axis=0)

        return y

    def my_collate(self,batch):
        """
        This function was created to handle a variable-length of the
        :param batch: tuple(data,target)
        :return: list[data_tensor(batch_size,n_samples_channels), target_tensor(batch_size,n_classes)]
        """
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]

        # define the max size of the batch
        m_size = 0
        for element in data:
            if m_size < element.shape[0]:
                m_size = element.shape[0]

        # zero pooling
        for index, element in enumerate(data):
            if m_size > element.shape[0]:
                padding = np.zeros((m_size-element.shape[0], element.shape[1]))
                padding = torch.from_numpy(padding)
                data[index] = torch.cat([element, padding], dim=0)
                padding = padding.detach()

        data = torch.stack(data)
        target = torch.stack(target)

        return [data, target]


class Dataset_test(Dataset_train):
    def __init__(self, patients):
        super().__init__(patients=patients)

    def __getitem__(self, idx):

        X,y = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X