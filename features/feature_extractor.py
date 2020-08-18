"""
feature_extractor.py
--------------------
This module provides a class and methods for pre-processing ECG signals and generating feature vectors using feature
extraction libraries.
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# 3rd party imports
import os
import time
import numpy as np
import pandas as pd
import simplejson as json
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

# Local imports
from kardioml import DATA_PATH, FS, ECG_LEADS
from kardioml.data.data_loader import parse_header
from kardioml.models.physionet2017.features.rri_features import RRIFeatures
from kardioml.models.physionet2017.features.template_features import TemplateFeatures
from kardioml.models.physionet2017.features.full_waveform_features import FullWaveformFeatures


class Features:
    def __init__(self, filename, waveform_data=None, header_data=None):

        # Set parameters
        self.filename = filename
        self.waveform_data = waveform_data
        self.header_data = header_data

        # Set attributes
        self.waveform_data = self._load_waveform_file()
        self.meta_data = self._import_meta_data()
        self.features = None

    def get_features(self):
        """Get features as Pandas DataFrame."""
        return pd.DataFrame([self.features])

    def save_features(self, lead):
        """Save features as JSON."""
        # Create directory for formatted data
        os.makedirs(os.path.join(DATA_PATH, 'physionet_2017', 'features', lead), exist_ok=True)

        # Save meta data JSON
        with open(
            os.path.join(DATA_PATH, 'physionet_2017', 'features', lead, '{}.json'.format(self.filename)), 'w'
        ) as file:
            json.dump(self.features, file, ignore_nan=True)

    def extract_features(
        self,
        lead,
        feature_groups,
        filter_bandwidth,
        normalize=True,
        polarity_check=True,
        template_before=0.2,
        template_after=0.4,
        show=False,
    ):

        # Get start time
        t_start = time.time()

        # Load .mat file
        signal_raw = self.waveform_data[ECG_LEADS.index(lead), :].flatten()

        # Preprocess signal
        ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates = self._preprocess_signal(
            signal_raw=signal_raw,
            filter_bandwidth=filter_bandwidth,
            normalize=normalize,
            polarity_check=polarity_check,
            template_before=template_before,
            template_after=template_after,
        )

        # Extract features from waveform
        self.features = self._group_features(
            ts=ts,
            signal_raw=signal_raw,
            signal_filtered=signal_filtered,
            rpeaks=rpeaks,
            templates_ts=templates_ts,
            templates=templates,
            template_before=template_before,
            template_after=template_after,
            feature_groups=feature_groups,
        )

        # Get end time
        t_end = time.time()

        # Print progress
        if show:
            print(
                'Finished extracting features from '
                + self.meta_data['filename']
                + ' | Extraction time: '
                + str(np.round((t_end - t_start) / 60, 3))
                + ' minutes'
            )

    def _import_meta_data(self):
        """Import meta data JSON files."""
        if self.header_data is None:
            return json.load(open(os.path.join(DATA_PATH, 'formatted', '{}.json'.format(self.filename))))
        else:
            meta_data = parse_header(self.header_data)
            self.filename = meta_data['filename']
            return meta_data

    def _load_waveform_file(self):
        """Loads ECG signal to numpy array from .mat file."""
        if self.waveform_data is None:
            return np.load(os.path.join(DATA_PATH, 'formatted', '{}.npy'.format(self.filename)))
        else:
            return self.waveform_data

    def _preprocess_signal(
        self, signal_raw, filter_bandwidth, normalize, polarity_check, template_before, template_after
    ):

        # Filter signal
        signal_filtered = self._apply_filter(signal_raw, filter_bandwidth)

        # Get BioSPPy ECG object
        ecg_object = ecg.ecg(signal=signal_raw, sampling_rate=FS, show=False)

        # Get BioSPPy output
        ts = ecg_object['ts']  # Signal time array
        rpeaks = ecg_object['rpeaks']  # rpeak indices

        # Get templates and template time array
        templates, rpeaks = self._extract_templates(signal_filtered, rpeaks, template_before, template_after)
        templates_ts = np.linspace(-template_before, template_after, templates.shape[1], endpoint=False)

        # Polarity check
        signal_raw, signal_filtered, templates = self._check_waveform_polarity(
            polarity_check=polarity_check,
            signal_raw=signal_raw,
            signal_filtered=signal_filtered,
            templates=templates,
        )
        # Normalize waveform
        signal_raw, signal_filtered, templates = self._normalize_waveform_amplitude(
            normalize=normalize, signal_raw=signal_raw, signal_filtered=signal_filtered, templates=templates
        )
        return ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates

    @staticmethod
    def _check_waveform_polarity(polarity_check, signal_raw, signal_filtered, templates):

        """Invert waveform polarity if necessary."""
        if polarity_check:

            # Get extremes of median templates
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):
                return signal_raw * -1, signal_filtered * -1, templates * -1
            else:
                return signal_raw, signal_filtered, templates

    @staticmethod
    def _normalize_waveform_amplitude(normalize, signal_raw, signal_filtered, templates):
        """Normalize waveform amplitude by the median R-peak amplitude."""
        if normalize:

            # Get median templates max
            templates_max = np.max(np.median(templates, axis=1))

            return signal_raw / templates_max, signal_filtered / templates_max, templates / templates_max

    @staticmethod
    def _extract_templates(signal_filtered, rpeaks, before, after):

        # convert delimiters to samples
        before = int(before * FS)
        after = int(after * FS)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    @staticmethod
    def _apply_filter(signal_raw, filter_bandwidth):
        """Apply FIR bandpass filter to waveform."""
        signal_filtered, _, _ = filter_signal(
            signal=signal_raw,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * FS),
            frequency=filter_bandwidth,
            sampling_rate=FS,
        )
        return signal_filtered

    def _group_features(
        self,
        ts,
        signal_raw,
        signal_filtered,
        rpeaks,
        templates_ts,
        templates,
        template_before,
        template_after,
        feature_groups,
    ):

        """Get a dictionary of all ECG features"""

        # Empty features dictionary
        features = dict()

        # Loop through feature groups
        for feature_group in feature_groups:

            # Full waveform features
            if feature_group == 'full_waveform_features':

                # Extract features
                full_waveform_features = FullWaveformFeatures(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=FS,
                )
                full_waveform_features.extract_full_waveform_features()

                # Update feature dictionary
                features.update(full_waveform_features.get_full_waveform_features())

            # RRI waveform features
            if feature_group == 'rri_features':

                # Extract features
                rri_features = RRIFeatures(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=FS,
                    template_before=template_before,
                    template_after=template_after,
                )
                rri_features.extract_rri_features()

                # Update feature dictionary
                features.update(rri_features.get_rri_features())

            # Template waveform features
            if feature_group == 'template_features':

                # Extract features
                template_features = TemplateFeatures(
                    ts=ts,
                    signal_raw=signal_raw,
                    signal_filtered=signal_filtered,
                    rpeaks=rpeaks,
                    templates_ts=templates_ts,
                    templates=templates,
                    fs=FS,
                    template_before=template_before,
                    template_after=template_after,
                )
                template_features.extract_template_features()

                # Update feature dictionary
                features.update(template_features.get_template_features())

            # Add age and sex
            features.update(
                {
                    'age': np.nan if self.meta_data['age'] == 'NaN' else int(self.meta_data['age']),
                    'sex': 1 if self.meta_data['sex'] else 0,
                }
            )

        return features
