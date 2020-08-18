"""
train.py
--------
This module provides a function for training a deep neural network in tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd part imports
import numpy as np
import tensorflow as tf

# Local imports
from kardioml.models.deepecg.train.logger import Logger
from kardioml.models.deepecg.train.monitor import Monitor
from kardioml.models.deepecg.train.summaries import Summaries
from kardioml.models.deepecg.utils.devices.device_check import get_device_count
from kardioml.models.deepecg.train.learning_rate_schedulers import AnnealingWarmRestartScheduler


def train(model, epochs, batch_size):
    """Trains a tf computational graph."""
    # Start Tensorflow session context
    with model.sess as sess:

        # Set seed
        tf.set_random_seed(seed=0)

        # Get number of GPUs
        num_gpus = get_device_count(device_type='GPU')

        # Get number of training batches
        num_train_batches = model.graph.generator_train.num_batches.eval(
            feed_dict={model.graph.batch_size: batch_size}
        )

        # Get number of batch steps per epoch
        steps_per_epoch = int(np.ceil(num_train_batches / num_gpus))

        # Initialize learning rate scheduler
        lr_scheduler = AnnealingWarmRestartScheduler(
            lr_min=1e-5,
            lr_max=1e-3,
            steps_per_epoch=steps_per_epoch,
            lr_max_decay=1.0,
            epochs_per_cycle=epochs,
            cycle_length_factor=1.5,
            warmup_factor=0.1,
        )

        # Initialize model model_tracker
        monitor = Monitor(
            sess=sess,
            graph=model.graph,
            learning_rate=lr_scheduler.lr,
            batch_size=batch_size,
            save_path=model.save_path,
            early_stopping_epoch=10,
            num_gpus=num_gpus,
        )

        # Initialize logger
        logger = Logger(
            monitor=monitor,
            epochs=epochs,
            save_path=model.save_path,
            log_epoch=1,
            batch_size=batch_size,
            num_train_batches=num_train_batches,
        )

        # Get mode handle for training
        handle_train = sess.run(model.graph.generator_train.iterator.string_handle())

        # Initialize summary writer
        summary_writer = Summaries(sess=sess, graph=model.graph, path=model.save_path)
        summary_writer.log_scalar_summaries(monitor=monitor)
        # summary_writer.log_val_cam_plots_summaries(monitor=monitor)

        # Loop through epochs
        for epoch in range(epochs):

            # Initialize metrics
            sess.run(fetches=[model.graph.init_metrics_op])

            # Loop through train dataset batches
            for batch in range(steps_per_epoch):

                # Run train operation
                _, _, train_summary, global_step = sess.run(
                    fetches=[
                        model.graph.train_op,
                        model.graph.update_metrics_op,
                        model.graph.train_summary_metrics_op,
                        model.graph.global_step,
                    ],
                    feed_dict={
                        model.graph.batch_size: batch_size,
                        model.graph.is_training: True,
                        model.graph.learning_rate: lr_scheduler.lr,
                        model.graph.mode_handle: handle_train,
                    },
                )

                # Add training summary
                summary_writer.log_train_summaries(summary=train_summary, global_step=global_step)

                # Update learning rate scheduler
                lr_scheduler.on_batch_end_update()

            # Initialize the train dataset iterator at the end of each epoch
            sess.run(
                fetches=[model.graph.generator_train.iterator.initializer],
                feed_dict={model.graph.batch_size: batch_size},
            )

            # Update monitor
            monitor.update_model_state(learning_rate=lr_scheduler.lr)

            # Log summaries
            summary_writer.log_val_scalar_summaries(monitor=monitor)

            # Log progress
            logger.log_training(monitor=monitor)

            # Check for early stopping
            if monitor.early_stopping_check() or epoch+1 == epochs:
                print('Early stopping at epoch {}'.format(epoch + 1))
                monitor.best_state.plot_val_cams()
                summary_writer.log_val_cam_plots_summaries(monitor=monitor)
                break

            # Update learning rate scheduler
            lr_scheduler.on_epoch_end_update()

        # End tracking
        monitor.end_monitoring()

        # End logging
        logger.end_log()

        # Close summary writers
        summary_writer.close_summaries()

    # Close tensorflow session
    model.close_session()
