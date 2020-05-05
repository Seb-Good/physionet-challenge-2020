"""
plot_formatted_data.py
----------------------
This module provides functions for plotting formatted data.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import os
import json
import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from ipywidgets import interact, fixed

# Local imports
from kardioml import DATA_PATH, FS, LABELS_LOOKUP, ECG_LEADS


def waveform_plot(filename_id, filenames, path):
    """Plot measure vs time."""
    # Get filename
    filename = filenames[filename_id]

    # Import waveforms
    hr = 0
    print('HR: {} BPM'.format(hr))
    waveforms = np.load(os.path.join(path, '{}.npy'.format(filename)))
    print('samples: {}'.format(waveforms.shape[1]))
    length = waveforms.shape[1] / FS
    print('FS: {} Hz'.format(FS))
    print('duration: {} seconds'.format(length))
    print('remainder: {} samples'.format(waveforms.shape[1] - 30000))
    print('remainder: {} seconds'.format(length - 60))

    hr_delta = 0
    print('HR Range: {} - {} BPM'.format(hr-hr_delta, hr+hr_delta))

    length_new = length - hr_delta / 60
    print('new duration: {} seconds'.format(length_new))
    samples_new = int(length_new * FS)
    print('new samples: {} samples'.format(samples_new))

    temp = signal.resample(waveforms, samples_new, axis=1)

    # length_new = 60
    # print('new duration: {} seconds'.format(length_new))
    # samples_new = int(length_new * FS)
    # print('new samples: {} seconds'.format(samples_new))
    # hr_new = hr * length_new / 60
    # print('new HR: {} BPM'.format(hr_new))
    # temp = signal.resample(waveforms, samples_new, axis=1)
    # temp_time = np.arange(length_new) * 1 / FS
    #
    # print(waveforms.shape)
    # print(temp.shape)

    # Import meta data
    meta_data = json.load(open(os.path.join(path, '{}.json'.format(filename))))

    # Get label
    label = ''
    for idx, lab in enumerate(meta_data['labels']):
        if idx == 0:
            label += LABELS_LOOKUP[lab]['label_full']
        else:
            label += ' and ' + LABELS_LOOKUP[lab]['label_full']

    # Time array
    time = np.arange(waveforms.shape[1]) * 1 / FS

    # Setup figure
    fig = plt.figure(figsize=(15, 15), facecolor='w')
    fig.subplots_adjust(wspace=0, hspace=0.05)
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # ECG
    ax1.set_title('File Name: {}\nAge: {}\nSex: {}\nLabel: {}'.format(filename,
                                                                      meta_data['age'],
                                                                      meta_data['sex'],
                                                                      label),
                  fontsize=20, loc='left',
                  x=0)
    shift = 0
    for channel_id in range(waveforms.shape[0]):
        ax1.plot(waveforms[channel_id, :] + shift, '-k', lw=2)
        ax1.plot(temp[channel_id, :] + shift, '-r', lw=2)
        ax1.text(0.1, 0.25 + shift, ECG_LEADS[channel_id], color='red', fontsize=16, ha='left')
        shift += 3

    # Axes labels
    ax1.set_xlabel('Time, seconds', fontsize=24)
    ax1.set_ylabel('ECG Amplitude, mV', fontsize=24)
    # ax1.set_xlim([time.min(), time.max()])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()


def waveform_plot_interact():
    """Launch interactive plotting widget."""
    # Set data path
    path = os.path.join(DATA_PATH, 'formatted')

    # Get filenames
    filenames = [filename.split('.')[0] for filename in os.listdir(path) if 'npy' in filename]

    interact(waveform_plot,
             filename_id=(0, len(filenames) - 1, 1),
             filenames=fixed(filenames),
             path=fixed(path))
