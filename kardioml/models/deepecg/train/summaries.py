"""
summaries.py
------------
This module provides a class and methods for writing training and validation summaries.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import os
import tensorflow as tf


class Summaries(object):

    """Writes training and validation summaries."""

    def __init__(self, sess, graph, path):

        # Set input parameters
        self.sess = sess
        self.graph = graph
        self.path = path

        # Set summary paths
        self.train_summary_path = os.path.join(self.path, 'train')
        self.val_summary_path = os.path.join(self.path, 'val')

        # Initialize Tensorboard writers
        self.train_summary_writer = tf.summary.FileWriter(
            logdir=self.train_summary_path, graph=self.sess.graph
        )
        self.val_summary_writer = tf.summary.FileWriter(logdir=self.val_summary_path)

    def log_train_summaries(self, summary, global_step):
        """Add training summary."""
        self.train_summary_writer.add_summary(summary=summary, global_step=global_step)

    def log_scalar_summaries(self, monitor):
        # Get training summary
        self.log_train_scalar_summaries(monitor=monitor)

        # Get validation summary
        self.log_val_scalar_summaries(monitor=monitor)

    def log_train_scalar_summaries(self, monitor):
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.train_loss)
        summary.value.add(tag='f_beta/f_beta', simple_value=monitor.current_state.train_f_beta)
        summary.value.add(tag='g_beta/g_beta', simple_value=monitor.current_state.train_g_beta)
<<<<<<< HEAD
        summary.value.add(tag='challenge_metric/challenge_metric',
                          simple_value=monitor.current_state.train_challenge_metric)
=======
        summary.value.add(
            tag='geometric_mean/geometric_mean', simple_value=monitor.current_state.train_geometric_mean
        )
>>>>>>> DS

        # Get validation summary
        self.train_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.train_summary_writer.flush()

    def log_val_scalar_summaries(self, monitor):
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.val_loss)
        summary.value.add(tag='f_beta/f_beta', simple_value=monitor.current_state.val_f_beta)
        summary.value.add(tag='g_beta/g_beta', simple_value=monitor.current_state.val_g_beta)
<<<<<<< HEAD
        summary.value.add(tag='challenge_metric/challenge_metric',
                          simple_value=monitor.current_state.val_challenge_metric)
=======
        summary.value.add(
            tag='geometric_mean/geometric_mean', simple_value=monitor.current_state.val_geometric_mean
        )
>>>>>>> DS

        # Get validation summary
        self.val_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.val_summary_writer.flush()

    def log_val_cam_plots_summaries(self, monitor):
        """Generate class activation map plot summaries."""
        # if monitor.current_state.val_geometric_mean == monitor.best_state.val_geometric_mean:

        # Get validation cam plots as numpy array
        val_cam_plots = self.sess.run([monitor.best_state.val_cam_plots])[0]

        # Get summary
        summary = self.sess.run(
            fetches=[self.graph.val_cam_plots_summary_op],
            feed_dict={self.graph.val_cam_plots: val_cam_plots},
        )

        # Write summary
        self.val_summary_writer.add_summary(summary=summary[0], global_step=monitor.best_state.global_step)

        # Flush summary writer
        self.val_summary_writer.flush()

    def close_summaries(self):
        """Close summary writers."""
        self.train_summary_writer.close()
        self.val_summary_writer.close()
