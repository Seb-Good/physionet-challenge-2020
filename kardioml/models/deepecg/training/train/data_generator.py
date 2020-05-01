"""
data_generator.py
-----------------
This module provides a class for generating data batches for training and evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import os
import json
import numpy as np
import tensorflow as tf

# Local imports
from kardioml import DATA_PATH, LABELS_COUNT


class DataGenerator(object):

    def __init__(self, path, mode, shape, batch_size, prefetch_buffer=1, seed=0, num_parallel_calls=1):

        # Set parameters
        self.path = path
        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.prefetch_buffer = prefetch_buffer
        self.seed = seed
        self.num_parallel_calls = num_parallel_calls

        # Set attributes
        self.file_names = self._get_lookup_dict()
        self.labels = self._get_labels()
        self.num_samples = len(self.file_names)
        self.file_paths = self._get_file_paths()
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.current_seed = 0

        # Get lambda functions
        self.import_waveforms_fn_train = lambda file_path, label: self._import_waveform(file_path=file_path,
                                                                                        label=label, augment=False)
        self.import_waveforms_fn_val = lambda file_path, label: self._import_waveform(file_path=file_path,
                                                                                      label=label, augment=False)
        # Get dataset
        self.dataset = self._get_dataset()

        # Get iterator
        self.iterator = self.dataset.make_initializable_iterator()

    def _get_next_seed(self):
        """update current seed"""
        self.current_seed += 1
        return self.current_seed

    def _get_file_paths(self):
        """Convert file names to full absolute file paths with .npy extension."""
        return [os.path.join(self.path, '{}.npy'.format(file_name )) for file_name in self.file_names]

    def _get_lookup_dict(self):
        """Load lookup dictionary {'train': ['A0001', ...], 'val': ['A0012', ...]}."""
        return json.load(open(os.path.join(DATA_PATH, 'deepecg', 'training_lookup.json')))[self.mode]

    def _get_labels(self):
        """Get list of waveform npy file paths and labels."""
        # Get label from each file
        labels = list()
        for filename in self.file_names:
            meta_data = json.load(open(os.path.join(self.path, '{}.json'.format(filename))))
            labels.append(meta_data['label_train'])

        # file_paths and labels should have same length
        assert len(self.file_names) == len(labels)

        return labels

    def _import_waveform(self, file_path, label, augment):
        """Import waveform file from file path string."""
        # Load numpy file
        waveform = tf.py_func(self._load_npy_file, [file_path], [tf.float32])

        # Zero pad
        waveform = tf.py_func(self._zero_pad, [waveform, 'center'], [tf.float32])

        # Set tensor shape
        waveform = tf.reshape(tensor=waveform, shape=self.shape)

        # Augment waveform
        if augment:
            waveform = self._augment(waveform=waveform)

        return waveform, label

    def _augment(self, waveform):
        """Apply random augmentations."""
        # Random amplitude scale
        waveform = self._random_scale(waveform=waveform, prob=0.5)

        # Random polarity flip
        waveform = self._random_polarity(waveform=waveform, prob=0.5)

        return waveform

    def _random_scale(self, waveform, prob):
        """Apply random multiplication factor."""
        # Get random true or false
        prediction = self._random_true_false(prob=prob)

        # Apply random multiplication factor
        waveform = tf.cond(prediction, lambda: self._scale(waveform=waveform),
                           lambda: self._do_nothing(waveform=waveform))

        return waveform

    @staticmethod
    def _scale(waveform):
        """Apply random multiplication factor."""
        # Get random scale factor
        scale_factor = tf.random_uniform(shape=[], minval=0.5, maxval=2.5, dtype=tf.float32)

        return waveform * scale_factor

    def _random_polarity(self, waveform, prob):
        """Apply random polarity flip."""
        # Get random true or false
        prediction = self._random_true_false(prob=prob)

        # Apply random polarity flip
        waveform = tf.cond(prediction, lambda: self._polarity(waveform=waveform),
                           lambda: self._do_nothing(waveform=waveform))

        return waveform

    @staticmethod
    def _polarity(waveform):
        """Apply random polarity flip."""
        return waveform * -1

    @staticmethod
    def _do_nothing(waveform):
        return waveform

    @staticmethod
    def _random_true_false(prob):
        """Get a random true or false."""
        # Get random probability between 0 and 1
        probability = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.less(x=probability, y=prob)

    def _get_dataset(self):
        """Retrieve tensorflow Dataset object."""
        if self.mode == 'train':
            return (
                tf.data.Dataset.from_tensor_slices(
                    tensors=(tf.constant(value=self.file_paths),
                             tf.reshape(tensor=tf.constant(self.labels), shape=[-1, LABELS_COUNT]))
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
                    tensors=(tf.constant(value=self.file_paths),
                             tf.reshape(tensor=tf.constant(self.labels), shape=[-1, LABELS_COUNT]))
                )
                .map(map_func=self.import_waveforms_fn_val, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )

    @staticmethod
    def _load_npy_file(file_path):
        """Python function for loading a single .npy file as casting the data type as float32."""
        return np.load(file_path.decode()).astype(np.float32)

    def _zero_pad(self, waveform, align):
        """Zero pad waveform (align: left, center, right)."""
        # Get remainder
        remainder = self.shape[0] - len(waveform)

        if align == 'left':
            return np.pad(waveform, (0, remainder), 'constant', constant_values=0)
        elif align == 'center':
            return np.pad(waveform, (int(remainder / 2), remainder - int(remainder / 2)), 'constant', constant_values=0)
        elif align == 'right':
            return np.pad(waveform, (remainder, 0), 'constant', constant_values=0)
