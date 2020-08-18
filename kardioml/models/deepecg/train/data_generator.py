"""
data_generator_pytorch.py
-----------------
This module provides a class for generating data batches for training and evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import os
import json
import random
import numpy as np
import tensorflow as tf
from scipy import signal
from biosppy.signals import ecg

# Local imports
from kardioml import LABELS_COUNT


class DataGenerator(object):
<<<<<<< HEAD

    def __init__(self, lookup_path, mode, shape, batch_size, fs,
                 prefetch_buffer=1, seed=0, num_parallel_calls=1):
=======
    def __init__(
        self,
        data_path,
        lookup_path,
        mode,
        shape,
        batch_size,
        fs=FS,
        prefetch_buffer=1,
        seed=0,
        num_parallel_calls=1,
    ):
>>>>>>> DS

        # Set parameters
        self.lookup_path = lookup_path
        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.fs = fs
        self.prefetch_buffer = prefetch_buffer
        self.seed = seed
        self.num_parallel_calls = num_parallel_calls

        # Set attributes
        self.file_names = self._get_lookup_dict()
        self.dataset = self._get_dataset()
        self.meta_data = self._get_meta_data()
        self.labels = self._get_labels()
        self.hr = self._get_hr()
        self.age = self._get_age()
        self.sex = self._get_sex()
        self.num_samples = len(self.file_names)
        self.waveform_file_paths = self._get_file_paths(file_type='npy')
        self.meta_file_paths = self._get_file_paths(file_type='json')
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.current_seed = 0

        # Get lambda functions
<<<<<<< HEAD
        self.import_waveforms_fn_train = \
            lambda waveform_file_path, meta_file_path, label, filename, dataset, hr, age, sex: self._import_waveform(
                waveform_file_path=waveform_file_path,
                meta_file_path=meta_file_path,
                label=label,
                filename=filename,
                dataset=dataset,
                hr=hr,
                age=age,
                sex=sex,
                augment=True
            )
        self.import_waveforms_fn_val = \
            lambda waveform_file_path, meta_file_path, label, filename, dataset, hr, age, sex: self._import_waveform(
                waveform_file_path=waveform_file_path,
                meta_file_path=meta_file_path,
                label=label,
                filename=filename,
                dataset=dataset,
                hr=hr,
                age=age,
                sex=sex,
                augment=False
            )
=======
        self.import_waveforms_fn_train = lambda waveform_file_path, meta_file_path, label, hr, age, sex: self._import_waveform(
            waveform_file_path=waveform_file_path,
            meta_file_path=meta_file_path,
            label=label,
            hr=hr,
            age=age,
            sex=sex,
            augment=True,
        )
        self.import_waveforms_fn_val = lambda waveform_file_path, meta_file_path, label, hr, age, sex: self._import_waveform(
            waveform_file_path=waveform_file_path,
            meta_file_path=meta_file_path,
            label=label,
            hr=hr,
            age=age,
            sex=sex,
            augment=False,
        )
>>>>>>> DS
        # Get dataset
        self.dataset = self._get_dataset()

        # Get iterator
        self.iterator = self.dataset.make_initializable_iterator()

    def _get_next_seed(self):
        """update current seed"""
        self.current_seed += 1
        return self.current_seed

    def _get_file_paths(self, file_type):
        """Convert file names to full absolute file paths with .npy extension."""
        return ['{}.{}'.format(file_name, file_type) for file_name in self.file_names]

    def _get_lookup_dict(self):
        """Load lookup dictionary {'train': ['A0001', ...], 'val': ['A0012', ...]}."""
        return json.load(open(os.path.join(self.lookup_path, 'training_lookup.json')))[self.mode]

    def _get_meta_data(self):
        """Import meta data JSONs."""
        meta_data = dict()
        for filename in self.file_names:
            meta_data[filename] = json.load(open('{}.json'.format(filename)))

        # file_paths and labels should have same length
        assert len(self.file_names) == len(meta_data)

        return meta_data

    def _get_labels(self):
        """Get list of waveform labels."""
        # Get label from each file
        labels = list()
        for filename in self.file_names:
            labels.append(self.meta_data[filename]['labels_training_merged'])

        # file_paths and labels should have same length
        assert len(self.file_names) == len(labels)

        return labels

    def _get_hr(self):
        """Get list of waveform heart rates."""
        # Get label from each file
        hr = list()
        for filename in self.file_names:
            hr.append(int(self.meta_data[filename]['hr']) if self.meta_data[filename]['hr'] != 'nan' else -1)

        # file_paths and labels should have same length
        assert len(self.file_names) == len(hr)

        return hr

    def _get_dataset(self):
        """Get list of datasets."""
        # Get label from each file
        dataset = list()
        for filename in self.file_names:
            dataset.append(self.meta_data[filename]['dataset'])

        # file_paths and labels should have same length
        assert len(self.file_names) == len(dataset)

        return dataset

    def _get_age(self):
        """Get list of waveform age."""
        # Get label from each file
        ages = list()
        for filename in self.file_names:
            ages.append(
                int(self.meta_data[filename]['age']) if self.meta_data[filename]['age'] != 'NaN' else -1
            )

        # file_paths and labels should have same length
        assert len(self.file_names) == len(ages)

        return ages

    def _get_sex(self):
        """Get list of waveform sex."""
        # Get sex from each file
        sexes = list()
        for filename in self.file_names:

            if self.meta_data[filename]['sex'] == 'male':
                sexes.append(1)
            elif self.meta_data[filename]['sex'] == 'female':
                sexes.append(0)
            else:
                sexes.append(-1)

        # file_paths and labels should have same length
        assert len(self.file_names) == len(sexes)

        return sexes

    @staticmethod
    def _import_hr(file_path):
        """Import meta data JSON."""
        meta_data = json.load(open(file_path))
        return int(meta_data['hr']) if meta_data['hr'] != 'nan' else -1

    def _import_waveform(self, waveform_file_path, meta_file_path, label, filename, dataset, hr, age, sex, augment):
        """Import waveform file from file path string."""
        # Load numpy file
        waveform = tf.py_func(self._load_npy_file, [waveform_file_path], tf.float32)

        # Resample waveform
        # waveform = tf.py_func(self._resample, [waveform], tf.float32)

        # Augment waveform
        if augment:

            # Random resample
            waveform = tf.py_func(self._random_resample, [waveform, hr], tf.float32)

            # Get rpeak channel
            rpeaks = tf.py_func(self._get_rpeak_channel, [waveform, meta_file_path], tf.float32)

            # Random amplitude scale
            waveform = self._random_scale(waveform=waveform, prob=0.75)

            # Apply synthetic noise
            waveform = tf.py_func(self._add_synthetic_noise, [waveform, 0.25], tf.float32)

            # Perturb age
            age = tf.py_func(self._random_age_perturbation, [age], tf.int32)

        else:
            # Get rpeak channel
            rpeaks = tf.py_func(self._get_rpeak_channel, [waveform, meta_file_path], tf.float32)

        # Combine waveform and rpeak
        # waveform = tf.concat([waveform, rpeaks], axis=1)

        # Pad waveform
        waveform = tf.py_func(self._pad_waveform, [waveform], tf.float32)

        # Set tensor shape
        waveform = tf.reshape(tensor=waveform, shape=self.shape)
        age = tf.reshape(tensor=age, shape=[1])

        return waveform, label, filename, dataset, age, sex

    def _get_rpeak_channel(self, waveform, meta_file_path):
        """Get binary array of rpeak locations."""
        waveform = waveform.squeeze()

        # Import meta data
        meta_data = json.load(open(meta_file_path))

        # Get time array
        time = np.arange(waveform.shape[0]) * 1 / self.fs

        # Create empty array with length of waveform
        rpeak_channel = np.zeros([waveform.shape[0], 1], dtype=np.float32)

        # Set rpeak sections
        for section in meta_data['rpeak_times']:
            rpeak_channel[np.where((time >= section[0]) & (time <= section[1]))[0]] = 1

        return rpeak_channel

    def _resample(self, waveform):
        """Randomly resample waveform."""
        if self.fs != FS:

            # Get waveform duration
            waveform = waveform.squeeze()

            # Get number of samples
            samples = int(waveform.shape[0] * self.fs / FS)

            # Resample waveform
            waveform = signal.resample_poly(waveform, samples, waveform.shape[0], axis=0).astype(np.float32)

            return waveform
        else:
            return waveform

    @staticmethod
    def _load_npy_file(file_path):
        """Python function for loading a single .npy file as casting the data type as float32."""
        return np.load(file_path.decode()).astype(np.float32)

    @staticmethod
    def _random_age_perturbation(age):
        """Randomly perturb age."""
        if age != -1:
            return int(age + np.random.uniform(-2, 2))
        else:
            return age

    def _pad_waveform(self, waveform):
        """Python function for padding waveform."""
        # Squeeze waveform
        waveform = waveform.squeeze()

        # Pad waveform
        remainder = self.shape[0] - waveform.shape[0]
        if remainder >= 0:
            return np.pad(
                waveform,
                ((int(remainder / 2), remainder - int(remainder / 2)), (0, 0)),
                'constant',
                constant_values=0,
            )
        else:
            return waveform[0 : self.shape[0], :]

    def _random_resample(self, waveform, hr):
        """Randomly resample waveform."""
        if hr != -1:
            # Get waveform duration
            waveform = waveform.squeeze()
            duration = waveform.shape[0] / self.fs

            # Get new heart rate
            hr_new = int(hr * np.random.uniform(0.9, 1.1))
            if hr_new > 300:
                hr_new = 300
            elif hr_new < 40:
                hr_new = 40
            else:
                pass

            # Get new duration
            duration_new = duration * hr / hr_new

            # Get number of samples
            samples = int(duration_new * self.fs)
            if samples > self.shape[0]:
                samples = self.shape[0]
            else:
                pass

            # Resample waveform
            waveform = signal.resample_poly(waveform, samples, waveform.shape[0], axis=0).astype(np.float32)

            return waveform

        else:
            return waveform

    def _random_scale(self, waveform, prob):
        """Apply random multiplication factor."""
        # Get random true or false
        prediction = self._random_true_false(prob=prob)

        # Apply random multiplication factor
        waveform = tf.cond(
            prediction, lambda: self._scale(waveform=waveform), lambda: self._do_nothing(waveform=waveform)
        )

        return waveform

    @staticmethod
    def _scale(waveform):
        """Apply random multiplication factor."""
        # Get random scale factor
        scale_factor = tf.random_uniform(shape=[], minval=0.25, maxval=3.0, dtype=tf.float32)

        return waveform * scale_factor

    @staticmethod
    def _do_nothing(waveform):
        return waveform

    def _add_synthetic_noise(self, waveform, probability=0.5):
        """Add different kinds of synthetic noise to the signal."""
        waveform = waveform.squeeze()
        for idx in range(waveform.shape[1]):
            waveform[:, idx] = self._generate_baseline_wandering_noise(
                waveform=waveform[:, idx], fs=self.fs, probability=probability
            )
            waveform[:, idx] = self._generate_high_frequency_noise(
                waveform=waveform[:, idx], fs=self.fs, probability=probability
            )
            waveform[:, idx] = self._generate_gaussian_noise(
                waveform=waveform[:, idx], probability=probability
            )
            waveform[:, idx] = self._generate_pulse_noise(waveform=waveform[:, idx], probability=probability)
        return waveform

    def _generate_baseline_wandering_noise(self, waveform, fs, probability=0.5):
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

    @staticmethod
    def _random_true_false(prob):
        """Get a random true or false."""
        # Get random probability between 0 and 1
        probability = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
        return tf.less(x=probability, y=prob)

    def _get_dataset(self):
        """Retrieve tensorflow Dataset object."""
        if self.mode == 'train':
            return (
                tf.data.Dataset.from_tensor_slices(
<<<<<<< HEAD
                    tensors=(tf.constant(value=self.waveform_file_paths),
                             tf.constant(value=self.meta_file_paths),
                             tf.reshape(tensor=tf.constant(value=self.labels), shape=[-1, LABELS_COUNT]),
                             tf.constant(value=self.file_names),
                             tf.constant(value=self.dataset),
                             tf.constant(value=self.hr),
                             tf.reshape(tf.constant(value=self.age), shape=[-1, 1]),
                             tf.reshape(tf.constant(value=self.sex), shape=[-1, 1]))
=======
                    tensors=(
                        tf.constant(value=self.waveform_file_paths),
                        tf.constant(value=self.meta_file_paths),
                        tf.reshape(tensor=tf.constant(value=self.labels), shape=[-1, LABELS_COUNT]),
                        tf.constant(value=self.hr),
                        tf.reshape(tf.constant(value=self.age), shape=[-1, 1]),
                        tf.reshape(tf.constant(value=self.sex), shape=[-1, 1]),
                    )
>>>>>>> DS
                )
                .shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
                .map(map_func=self.import_waveforms_fn_train, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )
        else:
            return (
                tf.data.Dataset.from_tensor_slices(
<<<<<<< HEAD
                    tensors=(tf.constant(value=self.waveform_file_paths),
                             tf.constant(value=self.meta_file_paths),
                             tf.reshape(tensor=tf.constant(value=self.labels), shape=[-1, LABELS_COUNT]),
                             tf.constant(value=self.file_names),
                             tf.constant(value=self.dataset),
                             tf.constant(value=self.hr),
                             tf.reshape(tf.constant(value=self.age), shape=[-1, 1]),
                             tf.reshape(tf.constant(value=self.sex), shape=[-1, 1]))
=======
                    tensors=(
                        tf.constant(value=self.waveform_file_paths),
                        tf.constant(value=self.meta_file_paths),
                        tf.reshape(tensor=tf.constant(value=self.labels), shape=[-1, LABELS_COUNT]),
                        tf.constant(value=self.hr),
                        tf.reshape(tf.constant(value=self.age), shape=[-1, 1]),
                        tf.reshape(tf.constant(value=self.sex), shape=[-1, 1]),
                    )
>>>>>>> DS
                )
                .map(map_func=self.import_waveforms_fn_val, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )
