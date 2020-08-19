# basic libs
import numpy as np
import json
import os
import gc
import random
from scipy import signal

# pytorch
import torch
from torch.utils.data import Dataset

#custom modules
from kardioml.data.resample import Resampling
from kardioml.data.p_t_wave_detection import PTWaveDetection
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, patients, aug,downsample):

        self.patients = patients
        self.aug = aug
        self.downsample=downsample

        self.resampling = Resampling()
        self.ptdetector = PTWaveDetection()

        self.preprocessing = Preprocessing(aug=aug)

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

        # TODO: FS experiment
        #data_folder = f'./data/{data_folder}/formatted/' #for tests

        # load waveforms
        #X = np.load(f'./data/{data_folder}/formatted/' + self.patients[id] + '.npy')
        X = np.load(f'./data/scipy_resample_1000_hz/{data_folder}/formatted/' + self.patients[id] + '.npy')





        # load annotation
        y = json.load(open(f'./data/scipy_resample_1000_hz/{data_folder}/formatted/' + self.patients[id] + '.json'))
        #y = json.load(open(f'./data/{data_folder}/formatted/' + self.patients[id] + '.json'))



        X,label = self.preprocessing.run(X=X,y=y)

        return X,label







    def my_collate(self, batch):
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
                padding = np.zeros((m_size - element.shape[0], element.shape[1]))
                padding = torch.from_numpy(padding)
                data[index] = torch.cat([element, padding], dim=0)
                padding = padding.detach()

        data = torch.stack(data)
        target = torch.stack(target)

        return [data, target]


class Dataset_test(Dataset_train):
    def __init__(self, patients):
        super().__init__(patients=patients,aug=False,downsample=False)

    def __getitem__(self, idx):

        X, y = self.load_data(idx, train=False)

        X = torch.tensor(X, dtype=torch.float)

        return X

class Preprocessing():

    def __init__(self,aug):

        self.aug = aug

    def run(self,X,y,label_process=True):

        if label_process:
            label = y['labels_training_merged']
            if label[4] > 0 or label[18] > 0:
                label[4] = 1
                label[18] = 1
            if label[23] > 0 or label[12] > 0:
                label[23] = 1
                label[12] = 1
            if label[26] > 0 or label[13] > 0:
                label[26] = 1
                label[13] = 1

        X = self.apply_amplitude_scaling(X=X, y=y)


        # add R, P, T waves
        r_waves = np.zeros((X.shape[0], 1))
        r_waves[y['rpeaks'][0], 0] = 1
        X = np.concatenate([X, r_waves], axis=1)




        if y['t_waves'] is None:
            X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        else:
            t_waves = y['t_waves'][0]
            t_waves_array = np.zeros((X.shape[0], 1))
            t_waves_array[t_waves, 0] = 1
            X = np.concatenate([X, t_waves_array], axis=1)


        if y['p_waves'] is None:
            X = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        else:
            p_waves = y['p_waves'][0]
            p_waves_array = np.zeros((X.shape[0], 1))
            p_waves_array[p_waves, 0] = 1
            X = np.concatenate([X, p_waves_array], axis=1)




        fs_training = 1000

        if self.aug is True:
            # pass
            X = self.apply_augmentation(waveform=X, meta_data=y, fs_training=fs_training)

        # padding
        sig_length = 19000

        if X.shape[0] < sig_length:
            padding = np.zeros((sig_length - X.shape[0], X.shape[1]))
            X = np.concatenate([X, padding], axis=0)
        if X.shape[0] > sig_length:
            X = X[:sig_length,:]

        if label_process:
            return X,label
        else:
            return X

    @staticmethod
    def apply_amplitude_scaling(X, y):
        """Get rpeaks for each channel and scale waveform amplitude by median rpeak amplitude of lead I."""
        if y['rpeaks']:
            for channel_rpeaks in y['rpeaks']:
                if channel_rpeaks:
                    return X / np.median(X[y['rpeaks'][0], 0])
        return (X - X.mean()) / X.std()

    def apply_augmentation(self, waveform, meta_data, fs_training):

        # Random resample
        # waveform = self._random_resample(waveform=waveform, meta_data=meta_data,
        #                                  fs_training=fs_training, probability=0.25)

        # Random amplitude scale
        waveform = self._random_scale(waveform=waveform, probability=0.5)

        # Apply synthetic noise
        waveform = self._add_synthetic_noise(waveform=waveform, fs_training=fs_training, probability=0.25)

        return waveform

    def _random_resample(self, waveform, meta_data, fs_training, probability):
        """Randomly resample waveform.
        bradycardia=3, sinus bradycardia=20, sinus tachycardia=22
        """
        if (
            meta_data['hr'] != 'nan'
            and all(meta_data['labels_training_merged'][label] == 0 for label in [3, 20, 22])
            and self._coin_flip(probability=probability)
        ):
            # Get waveform duration
            duration = waveform.shape[0] / fs_training

            # Get new heart rate
            hr_new = int(meta_data['hr'] * np.random.uniform(1, 1.25))
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

            # Resample waveform
            waveform = signal.resample_poly(waveform, samples, waveform.shape[0], axis=0).astype(np.float32)

            return waveform
        else:
            return waveform

    def _random_scale(self, waveform, probability):
        """Apply random scale factor between 0.25 and 3 to the waveform amplitudes."""
        # Get random scale factor
        scale_factor = random.uniform(0.25, 3.0)

        if self._coin_flip(probability):
            return waveform * scale_factor
        return waveform

    def _add_synthetic_noise(self, waveform, fs_training, probability):
        """Add different kinds of synthetic noise to the signal."""
        waveform = waveform.squeeze()
        for idx in range(waveform.shape[1]):
            waveform[:, idx] = self._generate_baseline_wandering_noise(
                waveform=waveform[:, idx], fs=fs_training, probability=probability
            )
            waveform[:, idx] = self._generate_high_frequency_noise(
                waveform=waveform[:, idx], fs=fs_training, probability=probability
            )
            waveform[:, idx] = self._generate_gaussian_noise(
                waveform=waveform[:, idx], probability=probability
            )
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
                waveform += random.uniform(0.01, 0.75) * np.sin(
                    2 * np.pi * random.uniform(0.001, 0.5) * time + random.uniform(0, 60)
                )

        return waveform

    def _generate_high_frequency_noise(self, waveform, fs, probability=0.5):
        """Adds high frequency sinusoidal noise to the input signal."""
        waveform = waveform.squeeze()
        if self._coin_flip(probability):
            # Generate time array
            time = np.arange(len(waveform)) * 1 / fs

            # Add noise
            waveform += random.uniform(0.001, 0.3) * np.sin(
                2 * np.pi * random.uniform(50, 200) * time + random.uniform(0, 60)
            )

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
            pulse = signal.gaussian(
                int(len(waveform) * random.uniform(0.05, 0.010)), std=random.randint(50, 200)
            )
            pulse = np.diff(pulse)

            # Get remainder
            remainder = len(waveform) - len(pulse)
            if remainder >= 0:
                left_pad = int(remainder * random.uniform(0.0, 1.0))
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