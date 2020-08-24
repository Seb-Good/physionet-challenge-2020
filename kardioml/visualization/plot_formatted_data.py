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
import matplotlib.pylab as plt
from ipywidgets import interact, fixed, IntSlider

# Local imports
from kardioml import WORKING_PATH, DATA_PATH, ECG_LEADS


def waveform_plot(filename_id, filenames, path, save_eps):
    """Plot measure vs time."""
    # Get filename
    filename = filenames[filename_id]

    # Import waveforms
    waveforms = np.load(os.path.join(path, '{}.npy'.format(filename)))

    # Import meta data
    meta_data = json.load(open(os.path.join(path, '{}.json'.format(filename))))

    # Scale waveforms
    waveforms = waveforms / np.median(waveforms[meta_data['rpeaks'][0], 0])

    # Get label
    label = ''
    if meta_data['labels_short']:
        for idx, lab in enumerate(meta_data['labels_full']):
            if idx == 0:
                label += lab
            else:
                label += ' and ' + lab
    else:
        label = 'Other'

    # Time array
    time = np.arange(waveforms.shape[0]) * 1 / meta_data['fs_resampled']

    # Setup figure
    fig = plt.figure(figsize=(15, 15), facecolor='w')
    fig.subplots_adjust(wspace=0, hspace=0.05)
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # ECG
    ax1.set_title(
        'File Name: {}\nAge: {}\nSex: {}\nLabel: {}\nHR: {} BPM'.format(
            filename, meta_data['age'], meta_data['sex'], label, int(meta_data['hr'])
        ),
        fontsize=20,
        loc='left',
        x=0,
    )
    shift = 0
    for channel_id in range(waveforms.shape[1]):
        ax1.plot(time, waveforms[:, channel_id] + shift, '-k', lw=2)
        if meta_data['rpeaks'] is not None:
            ax1.plot(
                time[meta_data['rpeaks'][channel_id]],
                waveforms[meta_data['rpeaks'][channel_id], channel_id] + shift,
                'ob',
            )
        if meta_data['p_and_t_waves']:
            if meta_data['p_waves'] is not None:
                ax1.plot(
                    time[meta_data['p_waves'][channel_id]],
                    waveforms[meta_data['p_waves'][channel_id], channel_id] + shift,
                    'or',
                )
            if meta_data['t_waves'] is not None:
                ax1.plot(
                    time[meta_data['t_waves'][channel_id]],
                    waveforms[meta_data['t_waves'][channel_id], channel_id] + shift,
                    'og',
                )
        ax1.text(0.1, 0.25 + shift, ECG_LEADS[channel_id], color='red', fontsize=16, ha='left')
        shift += 3
    ax1.plot(time, np.array(meta_data['rpeak_array']) + shift, '-k', lw=2)
    ax1.text(0.1, -1.25 + shift, 'R-Peaks', color='red', fontsize=16, ha='left')
    if meta_data['p_and_t_waves']:
        ax1.plot(time, np.array(meta_data['p_wave_array']) + shift + 3, '-k', lw=2)
        ax1.text(0.1, -1.25 + shift + 3, 'P-Waves', color='red', fontsize=16, ha='left')
        ax1.plot(time, np.array(meta_data['t_wave_array']) + shift + 6, '-k', lw=2)
        ax1.text(0.1, -1.25 + shift + 6, 'T-Waves', color='red', fontsize=16, ha='left')

    # Axes labels
    ax1.set_xlabel('Time, seconds', fontsize=24)
    ax1.set_ylabel('ECG Amplitude, mV', fontsize=24)
    ax1.set_xlim([time.min(), time.max()])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save_eps:
        os.makedirs(os.path.join(WORKING_PATH, 'figures'), exist_ok=True)
        plt.savefig(os.path.join(WORKING_PATH, 'figures', '12_lead_plot.eps'), format='eps')

    plt.show()


def waveform_plot_interact(dataset, save_eps=False):
    """Launch interactive plotting widget."""
    # Set data path
    path = os.path.join(DATA_PATH, dataset, 'formatted')

    # Get filenames
    filenames = [filename.split('.')[0] for filename in os.listdir(path) if 'npy' in filename]

    interact(
        waveform_plot,
        filename_id=IntSlider(value=0, min=0, max=len(filenames) - 1, step=1,),
        filenames=fixed(filenames),
        path=fixed(path),
        save_eps=fixed(save_eps)
    )
