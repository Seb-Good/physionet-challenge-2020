# -*- coding: utf-8 -*-
# pylint: disable-msg=

import numpy as np
from scipy import signal
from kardioml.segmentation.teijeiro.utils.units_helper import SAMPLING_FREQ
from kardioml.segmentation.teijeiro.model.interval import Interval as Iv


def filter_signal(x, fs, fmin, fmax):
    fmin = float(fmin)
    fmax = float(fmax)
    fs = float(fs)
    b, a = signal.butter(5, [2 * fmin / fs, 2 * fmax / fs], 'bandpass')
    return signal.filtfilt(b, a, x)


# assumes that x is an np.array
def reflect_signal(x, window_len):
    return np.concatenate(
        (np.concatenate((x[window_len:0:-1], x)), x[(len(x) - 2) : (len(x) - 2 - window_len) : -1])
    )


# returns a list with [time result_applying_fun_by_windows]
def reflect_and_rollapply(x, window_len, fun):
    if window_len % 2 == 0:
        reflect_len = window_len / 2
    else:
        reflect_len = (window_len - 1) / 2
    xr = reflect_signal(x, reflect_len)
    yr = rollapply(xr, window_len, fun, 1, True)
    # remove first reflect_len and last reflect_len samples, they are not 'real'
    return [range(len(x)), yr[1][reflect_len : (len(yr[0]) - reflect_len)]]


def rollapply(x, window_len, fun, window_shift=1, do_extend=False):
    ver_idx = np.arange(0, len(x) - window_len, window_shift)
    hor_idx = np.arange(window_len)
    hor_grid, ver_grid = np.meshgrid(hor_idx, ver_idx)
    idx_array = hor_grid + ver_grid
    x_array = x[idx_array]
    result = np.array([fun(x_array[i]) for i in range(len(x_array))])
    if do_extend and window_shift == 1:
        # First time step would be int(window_len / 2.0), so we have to add
        # int(window_len / 2.0) steps at the beginning
        # Last time step would be ver_idx[-1] + int(window_len / 2.0),
        # so we have to add len(x) - 1 - (ver_idx[-1] + int(window_len / 2.0))
        # + 1 samples at the end
        result = np.concatenate(
            (
                np.concatenate((np.repeat(result[0], int(window_len / 2.0) - 1), result)),
                np.repeat(result[-1], len(x) - ver_idx[-1] - int(window_len / 2.0)),
            )
        )
        # add time indices to the result
        result = [range(len(x)), result]
    else:
        # add time indices to the result
        result = [ver_idx + int(window_len / 2.0), result]
    return result


def is_valid_amplitude(x, min_limit, max_limit, min_range, max_range, tolerance, block_len):
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    if x_range == 0:
        return False
    if x_min < min_limit:
        return False
    if x_max > max_limit:
        return False
    if x_range < min_range:
        return False
    if x_range > max_range:
        return False

    # additional checks for large amplitude oscillations
    tl = x_min + tolerance * x_range
    th = x_min + (1 - tolerance) * x_range
    left_block = x[:block_len]
    right_block = x[block_len:]
    if (x_range > x_range / 10) and (
        (np.min(right_block) > th and np.max(left_block) < tl)
        or (np.min(left_block) > th and np.max(right_block) < tl)
    ):
        return False
    # avoid repeated values
    if np.min(left_block) == np.max(left_block) or np.min(right_block) == np.max(right_block):
        return False
    # signal oscillation between 2 values
    if np.percentile(x, 60) == x_max or np.percentile(x, 40) == x_min:
        return False

    return True


def is_valid_histogram(x):
    nbins = 10
    per = 1

    hist, _ = np.histogram(x, nbins)
    sum_perifery = hist[0] + hist[nbins - 1]
    sum_center = np.sum(hist[range(1, nbins - 1)])
    return sum_center > sum_perifery or hist[0] < sum_center * per or hist[nbins - 1] < sum_center * per


def mean_profile(x):
    return np.mean(np.abs(np.diff(x)))


def update_result(result, indices, win_len):
    half_win_len = int(win_len / 2.0)
    if win_len % 2 == 0:
        for i in indices:
            result[(i - half_win_len + 1) : (i + half_win_len + 1)] = True
    else:
        for i in indices:
            result[(i - half_win_len) : (i + half_win_len + 1)] = True


def get_episode_info(result):
    # find the start an end of episodes by using diff
    diff_result = np.diff(result.astype(int))
    episode_limits = [i for i, val in enumerate(diff_result) if val != 0]
    # There is an episode at the beginning of result.
    # Add an additional mark representing the beginning of this episode
    if result[0]:
        episode_limits.insert(0, -1)
    # There is an episode at the end of result.
    # Add an additional mark representing the end of this episode
    if result[-1]:
        episode_limits.append(len(episode_limits) - 1)
    # print jumps
    # differentiate again to get lengths of episodes and distances
    # between episodes:
    # even positions are the lengths of the episodes
    # odd positions are the distances between episodes
    distances = np.diff(episode_limits)
    return distances, episode_limits


def join_close_episodes(result, invalid_connect_len):
    episode_limits, jumps = get_episode_info(result)
    # odd positions are the distances between episodes
    for episode_number, distance in enumerate(episode_limits[1::2]):
        if distance < invalid_connect_len:
            interepisode_beg = jumps[2 * episode_number + 1]
            interepisode_end = jumps[2 * (episode_number + 1)] + 1
            result[interepisode_beg:interepisode_end] = True


def remove_small_episodes(result, minimum_episode_len):
    episode_limits, jumps = get_episode_info(result)
    # even positions are the lengths of the episodes
    for episode_number, episode_len in enumerate(episode_limits[0::2]):
        if episode_len < minimum_episode_len:
            episode_beg = jumps[2 * episode_number] + 1
            # Episodes at the beginning of the ECG are not removed!
            if episode_beg == 0:
                continue
            episode_end = jumps[2 * episode_number + 1] + 1
            result[episode_beg:episode_end] = False


def validity_test(tested_signal, min_value, max_value, min_range, max_range, min_range_large_window):
    profile_window = int(2 * SAMPLING_FREQ)
    profile_threshold = 0.02
    win_len = 240
    win_step = 240
    larger_win_len = 300
    larger_win_step = 120
    invalid_connect_len = int(0.8 * SAMPLING_FREQ)
    tolerance = 0.1
    block_len = 50
    minimum_episode_len = int(1.5 * SAMPLING_FREQ)
    # True means invalid signal
    result = np.repeat(False, len(tested_signal))

    hf_signal = filter_signal(tested_signal, SAMPLING_FREQ, 70.0, 90.0)

    # Amplitude-based tests
    test_indices = rollapply(
        tested_signal,
        larger_win_len,
        lambda x: (np.max(x) - np.min(x)) < min_range_large_window or np.isnan(x).any() or np.isinf(x).any(),
        larger_win_step,
    )
    update_result(result, test_indices[0][test_indices[1]], larger_win_len)

    invalid_indices = set()
    # Profile-based tests
    _, profile_signal = reflect_and_rollapply(tested_signal, profile_window, mean_profile)
    # plt.plot(tested_signal)
    # plt.plot(profile_signal * 100)
    result = np.logical_or(result, profile_signal > profile_threshold)

    # Amplitude-based tests with smaller window and additional tests
    test_indices = rollapply(
        tested_signal,
        win_len,
        lambda x: not is_valid_amplitude(
            x, min_value, max_value, min_range, max_range, tolerance, block_len
        ),
        win_step,
    )
    invalid_indices.update(set(test_indices[0][test_indices[1]].flatten()))
    # High-frequency test
    test_indices = rollapply(hf_signal, win_len, lambda x: np.min(np.abs(hf_signal)) > 0.005, win_step)
    invalid_indices.update(set(test_indices[0][test_indices[1]].flatten()))

    # Histogram based tests
    test_indices = rollapply(tested_signal, win_len, lambda (x): not is_valid_histogram(x), win_step)
    invalid_indices.update(set(test_indices[0][test_indices[1]].flatten()))

    update_result(result, invalid_indices, win_len)

    join_close_episodes(result, invalid_connect_len)

    remove_small_episodes(result, minimum_episode_len)

    return result


# input signal is supposed to be in physical units!!!
def detect_invalid_areas(input_signal):
    # return validity_test(input_signal, -7, 7, 0.005, 8, 0.02)
    return validity_test(input_signal, -4, 4, 0.005, 8, 0.02)


def get_intervals(invalid):
    """
    Returns noisy intervals as a sequence of tuples (begin, end), in samples.
    """
    if len(invalid) == 0:
        return []
    intervals = []
    tp = np.where(np.diff(invalid) != 0)[0]
    for i in range(len(tp)):
        if invalid[tp[i]]:
            beg = tp[i - 1] if i > 0 else 0
            intervals.append(Iv(beg, tp[i]))
    if invalid[-1]:
        beg = tp[-1] if len(tp) > 0 else 0
        intervals.append(Iv(beg, len(invalid) - 1))
    return intervals


def noise_autoxcorr(fragment):
    xcoefs = []
    if np.mean(fragment) > np.median(fragment):
        # Positive peaks
        tp = signal.argrelmax(fragment, order=50)[0]
        pc = np.percentile(fragment, range(50, 100))
        pkthres = pc[np.argmax(np.diff(pc)) + 1]
        tp = tp[fragment[tp] > pkthres]
    else:
        # Negative peaks
        tp = signal.argrelmin(fragment, order=50)[0]
        pc = np.percentile(fragment, range(1, 51))
        pkthres = pc[np.argmax(np.diff(pc)) + 1]
        tp = tp[fragment[tp] < pkthres]
    if len(tp) < 2:
        return 0.0
    for i in range(1, len(tp)):
        delay = tp[i] - tp[0]
        tr1, tr2 = fragment[:-delay], fragment[delay:]
        xcoefs.append((np.correlate(tr1, tr2) / np.sqrt(np.dot(tr1, tr1) * np.dot(tr2, tr2)))[0])
    return np.median(xcoefs)


def noise_intervals(signal):
    """
    Obtains a list of sorted intervals corresponding to noisy signal
    fragments. The signal must be in physical units (mV)
    """
    return get_intervals(detect_invalid_areas(signal))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import kardioml.segmentation.teijeiro.utils.MIT as MIT

    PATH = '/home/local/tomas.teijeiro/cinc_challenge17/sample/validation/'
    filename = 'A00005'
    series = MIT.load_MIT_record(PATH + filename, physical_units=True).signal[0]
    time = np.array(range(0, len(series))) / SAMPLING_FREQ
    invalid = detect_invalid_areas(series)
    plt.figure()
    plt.plot(time, series)
    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel('ECG (mV)')
    plt.plot(time, invalid)
    intervs = get_intervals(invalid)
    for iv in intervs:
        fragment = series[iv.start : iv.end + 1]
        xc = noise_autoxcorr(fragment)
        plt.text(time[iv.start], 1.05, '{0:.3f}'.format(xc))
    plt.show()
    # some basic testing for the filter_signal and rollapply based functions
    # t = np.linspace(-1, 1, 201)
    # fs = np.mean(1 / np.diff(t))
    # print fs
    # x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1)
    #                                            + 0.18*np.cos(2*np.pi*3.85*t))
    # xn = x + np.random.randn(len(t)) * 0.08
    # #plt.plot(xn)
    # y = filter_signal(xn, fs,3, 4.5)
    # plt.plot(y)
    # #plt.plot( 0.18*np.cos(2*np.pi*3.85*t))
    # #plt.show()
    #
    # plt.figure()
    # plt.plot(xn)
    # #detrended = rollapply(xn, 21, np.mean, do_extend=True)
    # #plt.plot(detrended[0],detrended[1])
    # detrended = reflect_and_rollapply(xn, 10, np.mean)
    # print len(xn)
    # print len(detrended[0])
    # plt.plot(detrended[0], detrended[1])
    # plt.show()
