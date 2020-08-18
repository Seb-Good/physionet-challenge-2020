"""
inference_data_loader.py
------------------------
This module provides classes and methods for formatting the Physionet2020 dataset for submission inference.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import numpy as np
from scipy import signal
from biosppy.signals import ecg
from scipy.signal.windows import blackmanharris

# Local imports
from kardioml.data.data_loader import parse_header
from kardioml.data.format_data_physionet_2020 import Labels
from kardioml.data.p_t_wave_detector import PTWaveDetection


def inference_data_loader(waveforms, header, fs_resampled, p_and_t_waves=False):
    """Convert data and header_data to .npy and dict format."""
    # Parse header data
    header = parse_header(header_data=header)

    # Transpose waveforms array
    waveforms = waveforms.T

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

    # Get meta data
    meta_data = {'filename': None,
                 'dataset': None,
                 'datetime': header['datetime'],
                 'channel_order': header['channel_order'],
                 'age': header['age'],
                 'sex': header['sex'],
                 'amp_conversion': header['amp_conversion'],
                 'fs': header['fs'],
                 'fs_resampled': fs_resampled,
                 'length': header['length'],
                 'num_leads': header['num_leads'],
                 'labels_SNOMEDCT': None,
                 'labels_short': None,
                 'labels_full': None,
                 'labels_int': None,
                 'labels_training': None,
                 'labels_training_merged': None,
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
                 'labels_unscored_SNOMEDCT': None,
                 'labels_unscored_short': None,
                 'labels_unscored_full': None,
                 'p_and_t_waves': p_and_t_waves}

    return waveforms, meta_data


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
