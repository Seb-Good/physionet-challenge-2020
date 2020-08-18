"""
state.py
--------
This module includes a model state class.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import os
import io
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from datetime import datetime
import matplotlib.pylab as plt
from scipy.special import expit
from scipy.stats.mstats import gmean

# Local imports
from metrics.metrics import Metric
from kardioml import DATA_PATH, WEIGHTS_PATH
from kardioml.scoring.scoring_metrics import (load_weights, compute_challenge_metric, compute_auc,
                                              compute_beta_measures, compute_f_measure, compute_accuracy)


class State(object):
    def __init__(self, sess, graph, save_path, learning_rate, batch_size, num_gpus):

        # Set input parameters
        self.sess = sess
        self.graph = graph
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        # Set attributes
        self.labels_scored_lookup = pd.read_csv(os.path.join(DATA_PATH, 'labels_scored.csv'))
        self.scoring_weights = load_weights(weight_file=WEIGHTS_PATH,
                                            classes=self.labels_scored_lookup['SNOMED CT Code'].values)
        self.train_loss = None
        self.val_loss = None
        self.train_f_beta = None
        self.val_f_beta = None
        self.train_g_beta = None
        self.val_g_beta = None
        self.train_challenge_metric = None
        self.val_challenge_metric = None
        self.logits = None
        self.labels = None
        self.file_names = None
        self.dataset = None
        self.waveforms = None
        self.cams = None
        self.val_cam_plots = None
        self.time = time.time()
        self.global_step = self._get_global_step()
        self.datetime = str(datetime.utcnow())
        self.num_train_batches = self._get_num_train_batches()
        self.num_val_batches = self._get_num_val_batches()
        self.train_steps_per_epoch = int(np.ceil(self.num_train_batches / self.num_gpus))
        self.val_steps_per_epoch = int(np.ceil(self.num_val_batches / self.num_gpus))
        self.epoch = self._get_epoch()

        # Compute training and validation metrics
        self._compute_metrics()

        # Get validation class activation map plots
        # self.val_cam_plots = self.plot_val_cams()

    def _compute_metrics(self):
        # Training metrics
<<<<<<< HEAD
        self.train_loss, self.train_f_beta, self.train_g_beta, self.train_challenge_metric = self._compute_train_metrics()

        # Validation metrics
        self.val_loss, self.val_f_beta, self.val_g_beta, self.val_challenge_metric = self._compute_val_metrics()
=======
        (
            self.train_loss,
            self.train_f_beta,
            self.train_g_beta,
            self.train_geometric_mean,
        ) = self._compute_train_metrics()

        # Validation metrics
        (
            self.val_loss,
            self.val_f_beta,
            self.val_g_beta,
            self.val_geometric_mean,
        ) = self._compute_val_metrics()
>>>>>>> DS

    def _get_num_train_batches(self):
        """Number of batches for training Dataset."""
        return self.graph.generator_train.num_batches.eval(
            feed_dict={self.graph.batch_size: self.batch_size}
        )

    def _get_num_val_batches(self):
        """Number of batches for validation Dataset."""
        return self.graph.generator_val.num_batches.eval(feed_dict={self.graph.batch_size: self.batch_size})

    def _get_global_step(self):
        return tf.train.global_step(self.sess, self.graph.global_step)

    def _get_epoch(self):
        return int(self.global_step / self.train_steps_per_epoch)

    def _compute_train_metrics(self):
        """Get training metrics."""
        if self.epoch > 0:
            # If metrics have been computed during a training epoch
            metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
            metrics = self.sess.run(metrics_op)

            return metrics['loss'], metrics['f_beta'], metrics['g_beta'], metrics['challenge_metric']

        else:
            # Get train handle
            handle_train = self.sess.run(self.graph.generator_train.iterator.string_handle())

            # Initialize train iterator
            self.sess.run(
                fetches=[self.graph.generator_train.iterator.initializer],
                feed_dict={self.graph.batch_size: self.batch_size},
            )

            # Initialize metrics
            self.sess.run(fetches=[self.graph.init_metrics_op])

            # Loop through train batches
            for batch in range(self.train_steps_per_epoch):

                # Run metric update operation
                self.sess.run(
                    fetches=[self.graph.update_metrics_op],
                    feed_dict={
                        self.graph.batch_size: self.batch_size,
                        self.graph.is_training: True,
                        self.graph.mode_handle: handle_train,
                    },
                )

            # Get metrics
            metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
            metrics = self.sess.run(metrics_op)

            return metrics['loss'], metrics['f_beta'], metrics['g_beta'], metrics['challenge_metric']

    def _compute_val_metrics(self):
        """Get validation metrics."""
        # Get val handle
        handle_val = self.sess.run(self.graph.generator_val.iterator.string_handle())

        # Initialize val iterator
        self.sess.run(
            fetches=[self.graph.generator_val.iterator.initializer],
            feed_dict={self.graph.batch_size: self.batch_size},
        )

        # Initialize metrics
        self.sess.run(fetches=[self.graph.init_metrics_op])

        # Empty lists for logits and labels
        logits_all = list()
        labels_all = list()
        dataset_all = list()
        file_names_all = list()
        waveforms_all = list()
        cams_all = list()

        # Loop through val batches
        for batch in range(self.val_steps_per_epoch):

            # Run metric update operation
<<<<<<< HEAD
            logits, labels, file_names, dataset, waveforms, cams, _ = self.sess.run(
                fetches=[self.graph.logits, self.graph.labels, self.graph.file_names, self.dataset,
                         self.graph.waveforms, self.graph.cams, self.graph.update_metrics_op],
                feed_dict={self.graph.batch_size: self.batch_size, self.graph.is_training: False,
                           self.graph.mode_handle: handle_val})
=======
            logits, labels, waveforms, cams, _ = self.sess.run(
                fetches=[
                    self.graph.logits,
                    self.graph.labels,
                    self.graph.waveforms,
                    self.graph.cams,
                    self.graph.update_metrics_op,
                ],
                feed_dict={
                    self.graph.batch_size: self.batch_size,
                    self.graph.is_training: False,
                    self.graph.mode_handle: handle_val,
                },
            )
>>>>>>> DS

            # Get logits and labels
            logits_all.append(logits)
            labels_all.append(labels)
            file_names_all.append(file_names)
            dataset_all.append(dataset)
            waveforms_all.append(waveforms)
            cams_all.append(cams)

        # Group logits and labels
        self.logits = np.concatenate(logits_all, axis=0)
        self.labels = np.concatenate(labels_all, axis=0)
        self.file_names = np.concatenate(file_names_all, axis=0)
        self.dataset = np.concatenate(dataset_all, axis=0)
        self.waveforms = np.concatenate(waveforms_all, axis=0)
        self.cams = np.concatenate(cams_all, axis=0)

<<<<<<< HEAD
        # Compute Beta-measures
        macro_f_beta_measure, macro_g_beta_measure = compute_beta_measures(
            labels=self.labels,
            outputs=np.round(expit(self.logits)).astype(int),
            beta=2
        )

        # Compute challenge metric
        challenge_metric = Metric().compute(labels=self.labels, outputs=np.round(expit(self.logits)).astype(int))
=======
        # Apply Normal Rhythm correction
        sigmoid = expit(self.logits)
        for index in range(sigmoid.shape[0]):
            if sigmoid[index, 3] >= 0.75 and np.argmax(sigmoid[index, :]) == 3:
                sigmoid[index, [0, 1, 2, 4, 5, 6, 7, 8]] = 0.0

        # Compute f1 score
        _, _, f_beta, g_beta = compute_beta_score(
            labels=self.labels,
            output=np.round(sigmoid).astype(int),
            beta=2,
            num_classes=LABELS_COUNT,
            check_errors=True,
        )
>>>>>>> DS

        # Get metrics
        metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
        metrics = self.sess.run(metrics_op)

        return metrics['loss'], macro_f_beta_measure, macro_g_beta_measure, challenge_metric

    def plot_val_cams(self):
        """Plot validation class activation maps."""
        # Empty list of cam plots as numpy arrays
        plots = list()

        # Loop through waveforms
        for index in range(64):

            # Get plot
            plot_buf = self._plot_image(index=index)

            # Convert plot to RGB array
            rgb_array = tf.image.decode_png(plot_buf.getvalue(), channels=4)

            # Append to list of plots
            plots.append(rgb_array)

        # Stack tensors along batch dimension
        batch_plot_tensor = tf.stack(plots, axis=0)

        self.val_cam_plots = batch_plot_tensor

    def _plot_image(self, index):

        # Setup figure
        fig = plt.figure(figsize=(20.0, 8.0), dpi=80)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        # Get time array
        time_array = np.arange(self.waveforms.shape[1]) * 1 / self.graph.network.hyper_params['fs']

        # Get labels
        label = self.labels[index]

        # Get logits
        logits = self.logits[index, :]

        # Get softmax
        sigmoid = expit(logits)

        # Get prediction
        prediction = int(np.squeeze(np.argmax(sigmoid)))

        # Get non-zero-pad indices
        non_zero_index = np.where(self.waveforms[index, :, 0] != 0)[0]

        # Title
<<<<<<< HEAD
        title_string = 'Label: {}\nPrediction: {}\n{}'
        ax1.set_title(title_string.format(label,
                                          np.round(expit(self.logits[index, :])).astype(int),
                                          np.round(expit(self.logits[index, :]), 2)),
                      fontsize=12, y=1.03)
=======
        title_string = '{}\nLabel: {}\nPrediction: {}\n{}'
        ax1.set_title(
            title_string.format(
                label_lookup,
                label,
                np.round(expit(self.logits[index, :])).astype(int),
                np.round(expit(self.logits[index, :]), 2),
            ),
            fontsize=20,
            y=1.03,
        )
>>>>>>> DS

        # Plot ECG waveform
        shift = 0
        for channel_id in range(self.waveforms.shape[2]):
            ax1.plot(
                time_array[non_zero_index], self.waveforms[index, non_zero_index, channel_id] + shift, '-k'
            )
            shift += 3
        ax1.set_xlim([time_array[non_zero_index].min(), time_array[non_zero_index].max()])
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        # ax1.set_ylabel('Normalized Amplitude', fontsize=22)
        # ax1.yaxis.set_tick_params(labelsize=16)

        # Plot Class Activation Map
        cams = signal.resample_poly(
            self.cams[index, :, :], self.graph.network.length, self.cams.shape[1], axis=0
        ).astype(np.float32)
        ax2.plot(time_array[non_zero_index], cams[non_zero_index, prediction], '-k')
        ax2.set_xlim([time_array[non_zero_index].min(), time_array[non_zero_index].max()])
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        # ax2.set_xlabel('Time, seconds', fontsize=22)
        # ax2.set_ylabel('Class Activation Map', fontsize=22)
        # ax2.xaxis.set_tick_params(labelsize=16)
        # ax2.yaxis.set_tick_params(labelsize=16)

        # Get image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    @staticmethod
    def _softmax(scores):
        """Compute softmax values for set of scores."""
        e_scores = np.exp(scores - np.max(scores))
        return e_scores / e_scores.sum()

    def create_cv_debug(self):
        """Generate CV Debug file."""
        predictions = np.round(expit(self.logits)).astype(int)
        for file_name, dataset, label, logit, prediction in zip(self.file_names.tolist(), self.dataset.tolist(),
                                                                self.labels.tolist(), self.logits.tolist(),
                                                                predictions.tolist()):
            # Save meta data JSON
            with open(os.path.join(self.save_path, 'cv_debug', dataset, '{}.json'.format(file_name)), 'w') as file:
                json.dump({'file_name': file_name,
                           'dataset': dataset,
                           'label': label,
                           'sigmoid': logit,
                           'predicted_label': prediction},
                          file, sort_keys=False, indent=4)
