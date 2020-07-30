"""
template_features.py
--------------------
This module provides a class and methods for extracting template features from ECG templates.
By: Sebastian D. Goodfellow, Ph.D.
"""

# 3rd party imports
import numpy as np
import scipy as sp
from pyentrp.entropy import sample_entropy

# Local imports
from kardioml.models.physionet2017.features.higuchi_fractal_dimension import hfd


class TemplateFeatures:

    """Extract template features for one ECG signal."""

    def __init__(
        self,
        ts,
        signal_raw,
        signal_filtered,
        rpeaks,
        templates_ts,
        templates,
        fs,
        template_before,
        template_after,
    ):

        # Input variables
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        self.template_before_ts = template_before
        self.template_after_ts = template_after
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)

        # Set future variables
        self.qrs_start_sp = None
        self.qrs_start_ts = None
        self.qrs_end_sp = None
        self.qrs_end_ts = None
        self.q_times_sp = None
        self.q_amps = None
        self.q_time_sp = None
        self.q_amp = None
        self.p_times_sp = None
        self.p_amps = None
        self.p_time_sp = None
        self.p_amp = None
        self.s_times_sp = None
        self.s_amps = None
        self.s_time_sp = None
        self.s_amp = None
        self.t_times_sp = None
        self.t_amps = None
        self.t_time_sp = None
        self.t_amp = None
        self.median_template = None

        # Set QRS start and end points
        self.qrs_start_sp_manual = 30  # R - qrs_start_sp_manual
        self.qrs_end_sp_manual = 40  # R + qrs_start_sp_manual

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # QRS calculations
        self.calculate_qrs_bounds()

        # PQRST Calculations
        self.preprocess_pqrst()

        # Feature dictionary
        self.template_features = dict()

    """
    Compile Features
    """

    def get_template_features(self):
        return self.template_features

    def extract_template_features(self):
        self.template_features.update(self.calculate_p_wave_features())
        self.template_features.update(self.calculate_q_wave_features())
        self.template_features.update(self.calculate_t_wave_features())
        self.template_features.update(self.calculate_s_wave_features())
        self.template_features.update(self.calculate_pqrst_wave_features())
        self.template_features.update((self.calculate_r_peak_polarity_features()))
        self.template_features.update(self.calculate_r_peak_amplitude_features())
        self.template_features.update(self.calculate_template_correlation_features())
        self.template_features.update(self.calculate_qrs_correlation_features())
        self.template_features.update(self.calculate_p_wave_correlation_features())
        self.template_features.update(self.calculate_t_wave_correlation_features())

    """
    Pre Processing
    """

    def calculate_qrs_bounds(self):

        # Empty lists of QRS start and end times
        qrs_starts_sp = []
        qrs_ends_sp = []

        # Loop through templates
        for template in range(self.templates.shape[1]):

            # Get zero crossings before the R-Peak
            pre_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates[0 : self.template_rpeak_sp, template]))
            )[0]

            # Check length
            if len(pre_qrs_zero_crossings) >= 2:

                # Append QRS starting index
                qrs_starts_sp = np.append(qrs_starts_sp, pre_qrs_zero_crossings[-2])

            if len(qrs_starts_sp) > 0:

                self.qrs_start_sp = int(np.median(qrs_starts_sp))
                self.qrs_start_ts = self.qrs_start_sp / self.fs

            else:
                self.qrs_start_sp = int(self.template_before_sp / 2.0)

            # Get zero crossings after the R-Peak
            post_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates[self.template_rpeak_sp : -1, template]))
            )[0]

            # Check length
            if len(post_qrs_zero_crossings) >= 2:

                # Append QRS ending index
                qrs_ends_sp = np.append(qrs_ends_sp, post_qrs_zero_crossings[-2])

            if len(qrs_ends_sp) > 0:

                self.qrs_end_sp = int(self.template_before_sp + np.median(qrs_ends_sp))
                self.qrs_end_ts = self.qrs_end_sp / self.fs

            else:
                self.qrs_end_sp = int(self.template_before_sp + self.template_after_sp / 2.0)

    def preprocess_pqrst(self):

        # Get QRS start point
        qrs_start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Get QRS end point
        qrs_end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # Get QR median template
        qr_median_template = self.median_template[qrs_start_sp : self.template_rpeak_sp]

        # Get RS median template
        rs_median_template = self.median_template[self.template_rpeak_sp : qrs_end_sp]

        # Get QR templates
        qr_templates = self.templates[qrs_start_sp : self.template_rpeak_sp, :]

        # Get RS templates
        rs_templates = self.templates[self.template_rpeak_sp : qrs_end_sp, :]

        """
        Q-Wave
        """
        # Get array of Q-wave times (sp)
        self.q_times_sp = np.array(
            [qrs_start_sp + np.argmin(qr_templates[:, col]) for col in range(qr_templates.shape[1])]
        )

        # Get array of Q-wave amplitudes
        self.q_amps = np.array(
            [self.templates[self.q_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.q_time_sp = qrs_start_sp + np.argmin(qr_median_template)

        # Get array of Q-wave amplitudes
        self.q_amp = self.median_template[self.q_time_sp]

        """
        P-Wave
        """
        # Get array of Q-wave times (sp)
        self.p_times_sp = np.array(
            [
                np.argmax(self.templates[0 : self.q_times_sp[col], col])
                for col in range(self.templates.shape[1])
            ]
        )

        # Get array of Q-wave amplitudes
        self.p_amps = np.array(
            [self.templates[self.p_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.p_time_sp = np.argmax(self.median_template[0 : self.q_time_sp])

        # Get array of Q-wave amplitudes
        self.p_amp = self.median_template[self.p_time_sp]

        """
        S-Wave
        """
        # Get array of Q-wave times (sp)
        self.s_times_sp = np.array(
            [
                self.template_rpeak_sp + np.argmin(rs_templates[:, col])
                for col in range(rs_templates.shape[1])
            ]
        )

        # Get array of Q-wave amplitudes
        self.s_amps = np.array(
            [self.templates[self.s_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.s_time_sp = self.template_rpeak_sp + np.argmin(rs_median_template)

        # Get array of Q-wave amplitudes
        self.s_amp = self.median_template[self.s_time_sp]

        """
        T-Wave
        """
        # Get array of Q-wave times (sp)
        self.t_times_sp = np.array(
            [
                self.s_times_sp[col] + np.argmax(self.templates[self.s_times_sp[col] :, col])
                for col in range(self.templates.shape[1])
            ]
        )

        # Get array of Q-wave amplitudes
        self.t_amps = np.array(
            [self.templates[self.t_times_sp[col], col] for col in range(self.templates.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.t_time_sp = self.s_time_sp + np.argmax(self.median_template[self.s_time_sp :])

        # Get array of Q-wave amplitudes
        self.t_amp = self.median_template[self.t_time_sp]

        """
        Debug
        """
        # import matplotlib.pylab as plt
        # plt.plot(self.templates, '-', c=[0.7, 0.7, 0.7])
        # plt.plot(self.median_template, '-k')
        # plt.plot([qrs_start_sp, qrs_start_sp], [self.median_template.min(), self.median_template.max()], '-r')
        # plt.plot([qrs_end_sp, qrs_end_sp], [self.median_template.min(), self.median_template.max()], '-r')
        #
        # plt.plot(self.q_times_sp, self.q_amps, '.r')
        # plt.plot(self.q_time_sp, self.q_amp, '.b', ms=10)
        #
        # plt.plot(self.p_times_sp, self.p_amps, '.r')
        # plt.plot(self.p_time_sp, self.p_amp, '.b', ms=10)
        #
        # plt.plot(self.s_times_sp, self.s_amps, '.r')
        # plt.plot(self.s_time_sp, self.s_amp, '.b', ms=10)
        #
        # plt.plot(self.t_times_sp, self.t_amps, '.r')
        # plt.plot(self.t_time_sp, self.t_amp, '.b', ms=10)
        #
        # plt.plot([self.p_time_sp-10, self.p_time_sp+10], [0, 0], '-g')
        # plt.plot([self.t_time_sp-10, self.t_time_sp+10], [0, 0], '-g')
        #
        # plt.ylim([self.median_template.min(), self.median_template.max()])
        #
        # plt.show()

    """
    Feature Methods
    """

    @staticmethod
    def safe_check(value):

        try:
            if np.isfinite(value):
                return value
            else:
                return np.nan

        except Exception:
            return np.nan

    def calculate_p_wave_features(self):

        # Empty dictionary
        p_wave_features = dict()

        # Get P-Wave energy bounds
        p_eng_start = self.p_time_sp - 10
        if p_eng_start < 0:
            p_eng_start = 0
        p_eng_end = self.p_time_sp + 10

        # Get end points
        start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Calculate p-wave statistics
        p_wave_features['p_wave_time'] = self.p_time_sp * 1 / self.fs
        p_wave_features['p_wave_time_std'] = np.std(self.p_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
        p_wave_features['p_wave_amp'] = self.p_amp
        p_wave_features['p_wave_amp_std'] = np.std(self.p_amps, ddof=1)
        p_wave_features['p_wave_eng'] = np.sum(np.power(self.median_template[p_eng_start:p_eng_end], 2))

        """
        Calculate non-linear statistics
        """
        entropy = [
            self.safe_check(
                sample_entropy(
                    self.templates[0:start_sp, col],
                    sample_length=2,
                    tolerance=0.1 * np.std(self.templates[0:start_sp, col]),
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        p_wave_features['p_wave_entropy_mean'] = np.mean(entropy)
        p_wave_features['p_wave_entropy_std'] = np.std(entropy, ddof=1)

        higuchi_fractal = [
            hfd(self.templates[0:start_sp, col], k_max=10) for col in range(self.templates.shape[1])
        ]
        p_wave_features['p_wave_higuchi_fractal_mean'] = np.mean(higuchi_fractal)
        p_wave_features['p_wave_higuchi_fractal_mean'] = np.mean(higuchi_fractal)
        p_wave_features['p_wave_higuchi_fractal_std'] = np.std(higuchi_fractal, ddof=1)

        return p_wave_features

    def calculate_q_wave_features(self):

        # Empty dictionary
        q_wave_features = dict()

        # Calculate p-wave statistics
        q_wave_features['q_wave_time'] = self.q_time_sp * 1 / self.fs
        q_wave_features['q_wave_time_std'] = np.std(self.q_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
        q_wave_features['q_wave_amp'] = self.q_amp
        q_wave_features['q_wave_amp_std'] = np.std(self.q_amps, ddof=1)

        return q_wave_features

    def calculate_s_wave_features(self):

        # Empty dictionary
        s_wave_features = dict()

        # Calculate p-wave statistics
        s_wave_features['s_wave_time'] = self.s_time_sp * 1 / self.fs
        s_wave_features['s_wave_time_std'] = np.std(self.s_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
        s_wave_features['s_wave_amp'] = self.s_amp
        s_wave_features['s_wave_amp_std'] = np.std(self.s_amps, ddof=1)

        return s_wave_features

    def calculate_t_wave_features(self):

        # Empty dictionary
        t_wave_features = dict()

        # Get T-Wave energy bounds
        t_eng_start = self.t_time_sp - 10
        t_eng_end = self.t_time_sp + 10
        if t_eng_end > self.templates.shape[0] - 1:
            t_eng_end = self.templates.shape[0] - 1

        # Get end points
        end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate p-wave statistics
        t_wave_features['t_wave_time'] = self.t_time_sp * 1 / self.fs
        t_wave_features['t_wave_time_std'] = np.std(self.t_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
        t_wave_features['t_wave_amp'] = self.t_amp
        t_wave_features['t_wave_amp_std'] = np.std(self.t_amps, ddof=1)
        t_wave_features['t_wave_eng'] = np.sum(np.power(self.median_template[t_eng_start:t_eng_end], 2))

        """
        Calculate non-linear statistics
        """
        entropy = [
            self.safe_check(
                sample_entropy(
                    self.templates[end_sp:, col],
                    sample_length=2,
                    tolerance=0.1 * np.std(self.templates[end_sp:, col]),
                )[0]
            )
            for col in range(self.templates.shape[1])
        ]
        t_wave_features['t_wave_entropy_mean'] = np.mean(entropy)
        t_wave_features['t_wave_entropy_std'] = np.std(entropy, ddof=1)

        higuchi_fractal = [
            hfd(self.templates[end_sp:, col], k_max=10) for col in range(self.templates.shape[1])
        ]
        t_wave_features['t_wave_higuchi_fractal_mean'] = np.mean(higuchi_fractal)
        t_wave_features['t_wave_higuchi_fractal_std'] = np.std(higuchi_fractal, ddof=1)

        return t_wave_features

    def calculate_pqrst_wave_features(self):

        # Empty dictionary
        pqrst_wave_features = dict()

        # PQ time
        pqi = (self.q_times_sp - self.p_times_sp) * 1 / self.fs
        pqrst_wave_features['pq_time'] = (self.q_time_sp - self.p_time_sp) * 1 / self.fs
        pqrst_wave_features['pq_time_std'] = np.std(pqi, ddof=1)

        # PR time
        pri = (self.template_rpeak_sp - self.p_times_sp) * 1 / self.fs
        pqrst_wave_features['pr_time'] = (self.template_rpeak_sp - self.p_time_sp) * 1 / self.fs
        pqrst_wave_features['pr_time_std'] = np.std(pri, ddof=1)

        # QR time
        qri = (self.template_rpeak_sp - self.q_times_sp) * 1 / self.fs
        pqrst_wave_features['qr_time'] = (self.template_rpeak_sp - self.q_time_sp) * 1 / self.fs
        pqrst_wave_features['qr_time_std'] = np.std(qri, ddof=1)

        # RS time
        rsi = (self.s_times_sp - self.template_rpeak_sp) * 1 / self.fs
        pqrst_wave_features['rs_time'] = (self.s_time_sp - self.template_rpeak_sp) * 1 / self.fs
        pqrst_wave_features['rs_time_std'] = np.std(rsi, ddof=1)

        # QS time
        qsi = (self.s_times_sp - self.q_times_sp) * 1 / self.fs
        pqrst_wave_features['qs_time'] = (self.s_time_sp - self.q_time_sp) * 1 / self.fs
        pqrst_wave_features['qs_time_std'] = np.std(qsi, ddof=1)

        # ST time
        sti = (self.t_times_sp - self.s_times_sp) * 1 / self.fs
        pqrst_wave_features['st_time'] = (self.t_time_sp - self.s_time_sp) * 1 / self.fs
        pqrst_wave_features['st_time_std'] = np.std(sti, ddof=1)

        # RT time
        rti = (self.t_times_sp - self.template_rpeak_sp) * 1 / self.fs
        pqrst_wave_features['rt_time'] = (self.t_time_sp - self.template_rpeak_sp) * 1 / self.fs
        pqrst_wave_features['rt_time_std'] = np.std(rti, ddof=1)

        # QT time
        qti = (self.t_times_sp - self.q_times_sp) * 1 / self.fs
        pqrst_wave_features['qt_time'] = (self.t_time_sp - self.q_time_sp) * 1 / self.fs
        pqrst_wave_features['qt_time_std'] = np.std(qti, ddof=1)

        # PT time
        pti = (self.t_times_sp - self.p_times_sp) * 1 / self.fs
        pqrst_wave_features['pt_time'] = (self.t_time_sp - self.p_time_sp) * 1 / self.fs
        pqrst_wave_features['pt_time_std'] = np.std(pti, ddof=1)

        # QRS energy
        pqrst_wave_features['qrs_energy'] = np.sum(
            np.power(self.median_template[self.q_time_sp : self.s_time_sp], 2)
        )

        return pqrst_wave_features

    def calculate_r_peak_polarity_features(self):

        # Empty dictionary
        r_peak_polarity_features = dict()

        # Get positive R-Peak amplitudes
        r_peak_positive = self.templates[self.template_rpeak_sp, :] > 0

        # Get negative R-Peak amplitudes
        r_peak_negative = self.templates[self.template_rpeak_sp, :] < 0

        # Calculate polarity statistics
        r_peak_polarity_features['positive_r_peaks'] = np.sum(r_peak_positive) / self.templates.shape[1]
        r_peak_polarity_features['negative_r_peaks'] = np.sum(r_peak_negative) / self.templates.shape[1]

        return r_peak_polarity_features

    def calculate_r_peak_amplitude_features(self):

        r_peak_amplitude_features = dict()

        rpeak_indices = self.rpeaks
        rpeak_amplitudes = self.signal_filtered[rpeak_indices]

        # Basic statistics
        r_peak_amplitude_features['rpeak_min'] = np.min(rpeak_amplitudes)
        r_peak_amplitude_features['rpeak_max'] = np.max(rpeak_amplitudes)
        r_peak_amplitude_features['rpeak_mean'] = np.mean(rpeak_amplitudes)
        r_peak_amplitude_features['rpeak_std'] = np.std(rpeak_amplitudes, ddof=1)
        r_peak_amplitude_features['rpeak_skew'] = sp.stats.skew(rpeak_amplitudes)
        r_peak_amplitude_features['rpeak_kurtosis'] = sp.stats.kurtosis(rpeak_amplitudes)

        # Non-linear statistics
        r_peak_amplitude_features['rpeak_entropy'] = self.safe_check(
            sample_entropy(rpeak_amplitudes, sample_length=2, tolerance=0.1 * np.std(rpeak_amplitudes))[0]
        )
        r_peak_amplitude_features['rpeak_higuchi_fractal_dimension'] = hfd(rpeak_amplitudes, k_max=10)

        return r_peak_amplitude_features

    def calculate_template_correlation_features(self):

        # Empty dictionary
        template_correlation_features = dict()

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(np.transpose(self.templates))

        # Get upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1).flatten()

        # Get upper triangle index where values are not zero
        upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

        # Get upper triangle values where values are not zero
        upper_triangle = upper_triangle[upper_triangle_index]

        # Calculate correlation matrix statistics
        template_correlation_features['template_corr_coeff_mean'] = np.mean(upper_triangle)
        template_correlation_features['template_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        return template_correlation_features

    def calculate_p_wave_correlation_features(self):

        # Empty dictionary
        p_wave_correlation_features = dict()

        # Get start point
        start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(np.transpose(self.templates[0:start_sp, :]))

        # Get upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1).flatten()

        # Get upper triangle index where values are not zero
        upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

        # Get upper triangle values where values are not zero
        upper_triangle = upper_triangle[upper_triangle_index]

        # Calculate correlation matrix statistics
        p_wave_correlation_features['p_wave_corr_coeff_mean'] = np.mean(upper_triangle)
        p_wave_correlation_features['p_wave_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        return p_wave_correlation_features

    def calculate_qrs_correlation_features(self):

        # Empty dictionary
        qrs_correlation_features = dict()

        # Get start and end points
        start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual
        end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(np.transpose(self.templates[start_sp:end_sp, :]))

        # Get upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1).flatten()

        # Get upper triangle index where values are not zero
        upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

        # Get upper triangle values where values are not zero
        upper_triangle = upper_triangle[upper_triangle_index]

        # Calculate correlation matrix statistics
        qrs_correlation_features['qrs_corr_coeff_mean'] = np.mean(upper_triangle)
        qrs_correlation_features['qrs_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        return qrs_correlation_features

    def calculate_t_wave_correlation_features(self):

        # Empty dictionary
        t_wave_correlation_features = dict()

        # Get end point
        end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(np.transpose(self.templates[end_sp:, :]))

        # Get upper triangle
        upper_triangle = np.triu(correlation_matrix, k=1).flatten()

        # Get upper triangle index where values are not zero
        upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

        # Get upper triangle values where values are not zero
        upper_triangle = upper_triangle[upper_triangle_index]

        # Calculate correlation matrix statistics
        t_wave_correlation_features['t_wave_corr_coeff_mean'] = np.mean(upper_triangle)
        t_wave_correlation_features['t_wave_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        return t_wave_correlation_features
