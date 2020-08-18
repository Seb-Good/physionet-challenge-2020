"""
rri_features.py
---------------
This module provides a class and methods for extracting RRI features from ECG signals.
By: Sebastian D. Goodfellow, Ph.D., 2017
"""

# 3rd party imports
import numpy as np
import scipy as sp
from scipy import signal
from scipy import interpolate
from pyentrp.entropy import sample_entropy

# Local imports
from kardioml.models.physionet2017.features.higuchi_fractal_dimension import hfd


class RRIFeatures:

    """Extract RRI features for one ECG signal."""

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

        # Set parameters
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        self.template_before_ts = template_before
        self.template_after_ts = template_after

        # Set attributes
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)
        self.rri = None
        self.rri_ts = None
        self.diff_rri = None
        self.diff_rri_ts = None
        self.diff2_rri = None
        self.diff2_rri_ts = None
        self.templates_good = None
        self.templates_bad = None
        self.median_template = None
        self.median_template_good = None
        self.median_template_bad = None
        self.rpeaks_good = None
        self.rpeaks_bad = None

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # Correct R-Peak picks
        self.r_peak_check(correlation_threshold=0.9)

        # RR interval calculations
        self.rpeaks_ts = self.ts[self.rpeaks]
        self.calculate_rr_intervals(correlation_threshold=0.9)

        # Feature dictionary
        self.rri_features = dict()

    """
    Compile Features
    """

    def get_rri_features(self):
        return self.rri_features

    def extract_rri_features(self):
        self.rri_features.update(
            self.calculate_rri_temporal_features(self.rri, self.diff_rri, self.diff2_rri)
        )
        self.rri_features.update(
            self.calculate_rri_nonlinear_features(self.rri, self.diff_rri, self.diff2_rri)
        )
        self.rri_features.update(self.calculate_pearson_correlation_features(self.rri))
        self.rri_features.update(self.calculate_rri_spectral_features(self.rri, self.rri_ts))
        self.rri_features.update(self.calculate_rpeak_detection_features())

    """
    Pre Processing
    """

    def r_peak_check(self, correlation_threshold=0.9):

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

        # Loop through rpeaks
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25, template_id],
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Compute cross correlation
                cross_correlation = signal.correlate(
                    self.median_template[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25],
                    self.templates[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25, template_id],
                )

                # Correct rpeak
                rpeak_corrected = self.rpeaks[template_id] - (
                    np.argmax(cross_correlation)
                    - len(self.median_template[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25])
                )

                # Check to see if shifting the R-Peak improved the correlation coefficient
                if self.check_improvement(rpeak_corrected, correlation_threshold):

                    # Update rpeaks array
                    self.rpeaks[template_id] = rpeak_corrected

        # Re-extract templates
        self.templates, self.rpeaks = self.extract_templates(self.rpeaks)

        # Re-compute median template
        self.median_template = np.median(self.templates, axis=1)

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

    def extract_templates(self, rpeaks):

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(self.signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - self.template_before_sp
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + self.template_after_sp
            if b > length:
                break

            # Append template list
            templates.append(self.signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def check_improvement(self, rpeak_corrected, correlation_threshold):

        # Before R-Peak
        a = rpeak_corrected - self.template_before_sp

        # After R-Peak
        b = rpeak_corrected + self.template_after_sp

        if a >= 0 and b < len(self.signal_filtered):

            # Update template
            template_corrected = self.signal_filtered[a:b]

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25],
                template_corrected[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25],
            )

            # Check new correlation
            if correlation_coefficient[0, 1] >= correlation_threshold:
                return True
            else:
                return False
        else:
            return False

    def calculate_rr_intervals(self, correlation_threshold=0.9):

        # Get rpeaks is floats
        rpeaks = self.rpeaks.astype(float)

        # Loop through templates
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25 : self.template_rpeak_sp + 25, template_id],
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Remove rpeak
                rpeaks[template_id] = np.nan

        # RRI
        rri = np.diff(rpeaks) * 1 / self.fs
        rri_ts = rpeaks[0:-1] / self.fs + rri / 2

        # RRI Velocity
        diff_rri = np.diff(rri)
        diff_rri_ts = rri_ts[0:-1] + diff_rri / 2

        # RRI Acceleration
        diff2_rri = np.diff(diff_rri)
        diff2_rri_ts = diff_rri_ts[0:-1] + diff2_rri / 2

        # Drop rri, diff_rri, diff2_rri outliers
        self.rri = rri[np.isfinite(rri)]
        self.rri_ts = rri_ts[np.isfinite(rri_ts)]
        self.diff_rri = diff_rri[np.isfinite(diff_rri)]
        self.diff_rri_ts = diff_rri_ts[np.isfinite(diff_rri_ts)]
        self.diff2_rri = diff2_rri[np.isfinite(diff2_rri)]
        self.diff2_rri_ts = diff2_rri_ts[np.isfinite(diff2_rri_ts)]

        # Get good and bad rpeaks
        self.rpeaks_good = self.rpeaks[np.isfinite(rpeaks)]
        self.rpeaks_bad = self.rpeaks[~np.isfinite(rpeaks)]

        # Get good and bad
        self.templates_good = self.templates[:, np.where(np.isfinite(rpeaks))[0]]
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.templates_bad = self.templates[:, np.where(~np.isfinite(rpeaks))[0]]

        # Get median templates
        self.median_template_good = np.median(self.templates_good, axis=1)
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.median_template_bad = np.median(self.templates_bad, axis=1)

    @staticmethod
    def safe_check(value):
        """Check for finite value and replace with np.nan if does not exist."""
        try:
            if np.isfinite(value):
                return value
            else:
                return np.nan
        except ValueError:
            return np.nan

    """
    Feature Methods
    """

    def calculate_rri_temporal_features(self, rri, diff_rri, diff2_rri):

        # Empty dictionary
        rri_temporal_features = dict()

        # RR interval statistics
        if len(rri) > 0:
            rri_temporal_features['rri_min'] = np.min(rri)
            rri_temporal_features['rri_max'] = np.max(rri)
            rri_temporal_features['rri_mean'] = np.mean(rri)
            rri_temporal_features['rri_median'] = np.median(rri)
            rri_temporal_features['rri_std'] = np.std(rri, ddof=1)
            rri_temporal_features['rri_skew'] = sp.stats.skew(rri)
            rri_temporal_features['rri_kurtosis'] = sp.stats.kurtosis(rri)
            rri_temporal_features['rri_rms'] = np.sqrt(np.mean(np.power(rri, 2)))
        else:
            rri_temporal_features['rri_min'] = np.nan
            rri_temporal_features['rri_max'] = np.nan
            rri_temporal_features['rri_mean'] = np.nan
            rri_temporal_features['rri_median'] = np.nan
            rri_temporal_features['rri_std'] = np.nan
            rri_temporal_features['rri_skew'] = np.nan
            rri_temporal_features['rri_kurtosis'] = np.nan
            rri_temporal_features['rri_rms'] = np.nan

        # Differences between successive RR interval differences statistics
        if len(diff_rri) > 0:
            rri_temporal_features['diff_rri_min'] = np.min(diff_rri)
            rri_temporal_features['diff_rri_max'] = np.max(diff_rri)
            rri_temporal_features['diff_rri_mean'] = np.mean(diff_rri)
            rri_temporal_features['diff_rri_median'] = np.median(diff_rri)
            rri_temporal_features['diff_rri_std'] = np.std(diff_rri, ddof=1)
            rri_temporal_features['diff_rri_skew'] = sp.stats.skew(diff_rri)
            rri_temporal_features['diff_rri_kurtosis'] = sp.stats.kurtosis(diff_rri)
            rri_temporal_features['diff_rri_rms'] = np.sqrt(np.mean(np.power(diff_rri, 2)))
        else:
            rri_temporal_features['diff_rri_min'] = np.nan
            rri_temporal_features['diff_rri_max'] = np.nan
            rri_temporal_features['diff_rri_mean'] = np.nan
            rri_temporal_features['diff_rri_median'] = np.nan
            rri_temporal_features['diff_rri_std'] = np.nan
            rri_temporal_features['diff_rri_skew'] = np.nan
            rri_temporal_features['diff_rri_kurtosis'] = np.nan
            rri_temporal_features['diff_rri_rms'] = np.nan

        # Differences between successive RR intervals statistics
        if len(diff2_rri) > 0:
            rri_temporal_features['diff2_rri_min'] = np.min(diff2_rri)
            rri_temporal_features['diff2_rri_max'] = np.max(diff2_rri)
            rri_temporal_features['diff2_rri_mean'] = np.mean(diff2_rri)
            rri_temporal_features['diff2_rri_median'] = np.median(diff2_rri)
            rri_temporal_features['diff2_rri_std'] = np.std(diff2_rri, ddof=1)
            rri_temporal_features['diff2_rri_kurtosis'] = sp.stats.kurtosis(diff2_rri)
            rri_temporal_features['diff2_rri_rms'] = np.sqrt(np.mean(np.power(diff2_rri, 2)))
        else:
            rri_temporal_features['diff2_rri_min'] = np.nan
            rri_temporal_features['diff2_rri_max'] = np.nan
            rri_temporal_features['diff2_rri_mean'] = np.nan
            rri_temporal_features['diff2_rri_median'] = np.nan
            rri_temporal_features['diff2_rri_std'] = np.nan
            rri_temporal_features['diff2_rri_kurtosis'] = np.nan
            rri_temporal_features['diff2_rri_rms'] = np.nan

        # pNN statistics
        if len(diff_rri) > 0:
            rri_temporal_features['pnn20'] = self.pnn(diff_rri, 0.02)
            rri_temporal_features['pnn50'] = self.pnn(diff_rri, 0.05)
        else:
            rri_temporal_features['pnn20'] = np.nan
            rri_temporal_features['pnn50'] = np.nan

        return rri_temporal_features

    @staticmethod
    def consecutive_count(random_list):

        retlist = []
        count = 1
        # Avoid IndexError for  random_list[i+1]
        for i in range(len(random_list) - 1):
            # Check if the next number is consecutive
            if random_list[i] + 1 == random_list[i + 1]:
                count += 1
            else:
                # If it is not append the count and restart counting
                retlist = np.append(retlist, count)
                count = 1
        # Since we stopped the loop one early append the last count
        retlist = np.append(retlist, count)

        return retlist

    def calculate_rri_nonlinear_features(self, rri, diff_rri, diff2_rri):

        # Empty dictionary
        rri_nonlinear_features = dict()

        # Non-linear RR statistics
        if len(rri) > 1:
            rri_nonlinear_features['rri_entropy'] = self.safe_check(
                sample_entropy(rri, sample_length=2, tolerance=0.1 * np.std(rri))[0]
            )
            rri_nonlinear_features['rri_higuchi_fractal_dimension'] = self.safe_check(hfd(rri, k_max=10))
        else:
            rri_nonlinear_features['rri_entropy'] = np.nan
            rri_nonlinear_features['rri_higuchi_fractal_dimension'] = np.nan

        # Non-linear RR difference statistics
        if len(diff_rri) > 1:
            rri_nonlinear_features['diff_rri_entropy'] = self.safe_check(
                sample_entropy(diff_rri, sample_length=2, tolerance=0.1 * np.std(diff_rri))[0]
            )
            rri_nonlinear_features['diff_rri_higuchi_fractal_dimension'] = self.safe_check(
                hfd(diff_rri, k_max=10)
            )
        else:
            rri_nonlinear_features['diff_rri_entropy'] = np.nan
            rri_nonlinear_features['diff_rri_higuchi_fractal_dimension'] = np.nan

        # Non-linear RR difference difference statistics
        if len(diff2_rri) > 1:
            rri_nonlinear_features['diff2_rri_entropy'] = self.safe_check(
                sample_entropy(diff2_rri, sample_length=2, tolerance=0.1 * np.std(diff2_rri))[0]
            )
            rri_nonlinear_features['diff2_rri_higuchi_fractal_dimension'] = self.safe_check(
                hfd(diff2_rri, k_max=10)
            )
        else:
            rri_nonlinear_features['diff2_rri_entropy'] = np.nan
            rri_nonlinear_features['diff2_rri_higuchi_fractal_dimension'] = np.nan

        return rri_nonlinear_features

    @staticmethod
    def pnn(diff_rri, time):

        # Count number of rri diffs greater than the specified time
        nn = np.sum(np.abs(diff_rri) > time)

        # Compute pNN
        pnn = nn / len(diff_rri) * 100

        return pnn

    @staticmethod
    def calculate_pearson_correlation_features(rri):

        # Empty dictionary
        pearson_correlation_features = dict()

        if len(rri[0:-2]) == len(rri[1:-1]) and len(rri[0:-2]) > 2 and len(rri[0:-2]) > 2:

            # Calculate Pearson correlation
            pearson_coeff_p1, pearson_p_value_p1 = sp.stats.pearsonr(rri[0:-2], rri[1:-1])

            # Get features
            pearson_correlation_features['rri_p1_pearson_coeff'] = pearson_coeff_p1
            pearson_correlation_features['rri_p1_pearson_p_value'] = pearson_p_value_p1

        else:
            pearson_correlation_features['rri_p1_pearson_coeff'] = np.nan
            pearson_correlation_features['rri_p1_pearson_p_value'] = np.nan

        return pearson_correlation_features

    @staticmethod
    def calculate_rri_spectral_features(rri, rri_ts):

        # Empty dictionary
        rri_spectral_features = dict()

        if len(rri) > 3:

            # Zero the time array
            rri_ts = rri_ts - rri_ts[0]

            # Set resampling rate
            fs = 10  # Hz

            # Generate new resampling time array
            rri_ts_interp = np.arange(rri_ts[0], rri_ts[-1], 1 / float(fs))

            # Setup interpolation function
            tck = interpolate.splrep(rri_ts, rri, s=0)

            # Interpolate rri on new time array
            rri_interp = interpolate.splev(rri_ts_interp, tck, der=0)

            # Set frequency band limits [Hz]
            vlf_band = (0, 0.04)  # Very low frequency
            lf_band = (0.04, 0.15)  # Low frequency
            hf_band = (0.15, 0.6)  # High frequency
            vhf_band = (0.6, 2)  # High frequency

            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=rri_interp, fs=fs)

            # Get frequency band indices
            vlf_index = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
            lf_index = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
            hf_index = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])
            vhf_index = np.logical_and(fxx >= vhf_band[0], fxx < vhf_band[1])

            # Compute power in each frequency band
            vlf_power = np.trapz(y=pxx[vlf_index], x=fxx[vlf_index])
            lf_power = np.trapz(y=pxx[lf_index], x=fxx[lf_index])
            hf_power = np.trapz(y=pxx[hf_index], x=fxx[hf_index])
            vhf_power = np.trapz(y=pxx[vhf_index], x=fxx[vhf_index])

            # Compute total power
            total_power = vlf_power + lf_power + hf_power + vhf_power

            # Compute spectral ratios
            rri_spectral_features['rri_low_high_spectral_ratio'] = lf_power / hf_power
            rri_spectral_features['rri_low_very_high_spectral_ratio'] = lf_power / vhf_power
            rri_spectral_features['rri_low_frequency_power'] = (lf_power / total_power) * 100
            rri_spectral_features['rri_high_frequency_power'] = (hf_power / total_power) * 100
            rri_spectral_features['rri_very_high_frequency_power'] = (vhf_power / total_power) * 100
            rri_spectral_features['rri_freq_max_frequency_power'] = fxx[
                np.argmax(pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])])
            ]
            rri_spectral_features['rri_power_max_frequency_power'] = np.max(
                pxx[np.logical_and(fxx >= lf_band[0], fxx < vhf_band[1])]
            )

        else:
            # Compute spectral ratios
            rri_spectral_features['rri_low_high_spectral_ratio'] = np.nan
            rri_spectral_features['rri_low_very_high_spectral_ratio'] = np.nan
            rri_spectral_features['rri_low_frequency_power'] = np.nan
            rri_spectral_features['rri_high_frequency_power'] = np.nan
            rri_spectral_features['rri_very_high_frequency_power'] = np.nan
            rri_spectral_features['rri_freq_max_frequency_power'] = np.nan
            rri_spectral_features['rri_power_max_frequency_power'] = np.nan

        return rri_spectral_features

    def calculate_rpeak_detection_features(self):

        # Empty dictionary
        rpeak_detection_features = dict()

        # Get median rri
        if len(self.rri) > 0:

            # Compute median rri
            rri_avg = np.median(self.rri)

        else:

            # Define possible rri's
            th1 = 1.5  # 40 bpm
            th2 = 0.3  # 200 bpm

            # Compute mean rri
            rri_avg = (th1 + th2) / 2

        # Calculate waveform duration in seconds
        time_duration = np.max(self.ts)

        # Calculate theoretical number of expected beats
        beat_count_theory = np.ceil(time_duration / rri_avg)

        # Calculate percentage of observed beats to theoretical beats
        rpeak_detection_features['detection_success'] = len(self.rpeaks) / beat_count_theory

        # Calculate percentage of bad rpeaks
        if self.rpeaks_bad is None:
            rpeak_detection_features['rpeaks_rejected'] = 0.0
        else:
            rpeak_detection_features['rpeaks_rejected'] = len(self.rpeaks_bad) / len(self.rpeaks)

        return rpeak_detection_features
