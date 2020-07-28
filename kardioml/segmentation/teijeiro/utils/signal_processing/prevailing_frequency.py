import numpy as np


def get_prevailing_freq(y, fs=1.0):
    y_fft = np.fft.fft(y) / float(len(y))
    freq = np.fft.fftfreq(len(y))
    idx = freq >= 0.0
    spectrum = np.absolute(y_fft[idx]) ** 2
    max_idx = np.argmax(spectrum)
    prevailing_freq = (freq[idx])[max_idx]
    prevailing_amp = spectrum[max_idx]
    return prevailing_freq * fs, prevailing_amp
