"""
deepecg_v1.py
-------------
This module provides a class and methods for building a convolutional neural network with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# 3rd party imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports
from kardioml import DATA_PATH, WEIGHTS_PATH
from metrics.metrics import Metric
from kardioml.models.deepecg.train.data_generator import DataGenerator
<<<<<<< HEAD
from kardioml.models.deepecg.networks.layers import (fc_layer, conv_layer, dropout_layer,
                                                     print_output_shape, max_pool_layer)
from kardioml.scoring.scoring_metrics import (load_weights, compute_challenge_metric, compute_auc,
                                              compute_beta_measures, compute_f_measure, compute_accuracy)
=======
from kardioml.models.deepecg.networks.layers import (
    fc_layer,
    conv_layer,
    dropout_layer,
    print_output_shape,
    max_pool_layer,
)
>>>>>>> DS


class DeepECGV1(object):

    """
    Build the forward propagation computational graph for a WavNet inspired deep neural network.
    """

    def __init__(self, length, channels, classes, hyper_params, seed=0):

        # Set input parameters
        self.length = length
        self.channels = channels
        self.classes = classes
        self.hyper_params = hyper_params
        self.seed = seed
        self.labels_scored_lookup = pd.read_csv(os.path.join(DATA_PATH, 'labels_scored.csv'))
        self.scoring_weights = load_weights(weight_file=WEIGHTS_PATH,
                                            classes=self.labels_scored_lookup['SNOMED CT Code'].values.tolist())

    def inference(self, input_layer, age, sex, reuse, is_training, name, print_shape=True):
        """Forward propagation of computational graph."""
        # Check input layer dimensions
        assert input_layer.shape[1] == self.length
        assert input_layer.shape[2] == self.channels

        # Define a scope for reusing the variables
        with tf.variable_scope(name, reuse=reuse):

            # Set variables
            skips = list()

            # Print shape
            print_output_shape(layer_name='input', net=input_layer, print_shape=print_shape)

            """Stem Layers"""
            # --- Stem Layer 1 (Convolution) ------------------------------------------------------------------------- #

            # Set name
            layer_name = 'stem_layer_1'

            # Set layer scope
            with tf.variable_scope(layer_name):

                # Convolution
                net = conv_layer(
                    input_layer=input_layer,
                    kernel_size=self.hyper_params['kernel_size'],
                    strides=1,
                    dilation_rate=1,
                    filters=self.hyper_params['conv_filts'] / 2,
                    padding='SAME',
                    activation=tf.nn.relu,
                    use_bias=False,
                    name=layer_name + '_conv',
                    seed=self.seed,
                )

                # Max pool
                net = max_pool_layer(
                    input_layer=net, pool_size=3, strides=2, padding='SAME', name=layer_name + '_maxpool'
                )

                # Dropout
                net = dropout_layer(
                    input_layer=net,
                    drop_rate=self.hyper_params['drop_rate'],
                    seed=self.seed,
                    training=is_training,
                    name=layer_name + '_dropout',
                )

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Stem Layer 2 (Convolution) ------------------------------------------------------------------------- #

            # Set name
            layer_name = 'stem_layer_2'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(
                    input_layer=net,
                    kernel_size=self.hyper_params['kernel_size'],
                    strides=1,
                    dilation_rate=1,
                    filters=self.hyper_params['conv_filts'],
                    padding='SAME',
                    activation=tf.nn.relu,
                    use_bias=False,
                    name=layer_name + '_conv',
                    seed=self.seed,
                )

                # Max pool
                net = max_pool_layer(
                    input_layer=net, pool_size=3, strides=2, padding='SAME', name=layer_name + '_maxpool'
                )

                # Dropout
                net = dropout_layer(
                    input_layer=net,
                    drop_rate=self.hyper_params['drop_rate'],
                    seed=self.seed,
                    training=is_training,
                    name=layer_name + '_dropout',
                )

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)
            outputs = {'res': net}

            """Residual Layers"""
            for res_id in np.arange(1, self.hyper_params['num_res_layers'] + 1):

                # Set name
                layer_name = 'res_layer_{}'.format(res_id)

                # Set dilation rate
                if self.hyper_params['dilation']:
                    dilation_rate = int(2 ** res_id)
                else:
                    dilation_rate = 1

                # Set res out
                res = True if res_id != self.hyper_params['num_res_layers'] else False

                # Compute block
                outputs = self._residual_block(
                    input_layer=outputs['res'],
                    kernel_size=self.hyper_params['kernel_size'],
                    layer_name=layer_name,
                    conv_filts=self.hyper_params['conv_filts'],
                    res_filts=self.hyper_params['res_filts'],
                    skip_filts=self.hyper_params['skip_filts'],
                    is_training=is_training,
                    dilation_rate=dilation_rate,
                    res=res,
                    skip=True,
                )

                # Collect skip and res
                skips.append(outputs['skip'])

                # Print shape
                print_output_shape(
                    layer_name=layer_name + '_skip', net=outputs['skip'], print_shape=print_shape
                )

            # Add all skips to res output
            with tf.variable_scope('skips'):
                output = tf.add_n(inputs=skips, name='add_skips')

            # Print shape
            print_output_shape(layer_name='output_skip_addition', net=output, print_shape=print_shape)

            # Activation
            with tf.variable_scope('relu') as scope:
                output = tf.nn.relu(output, name=scope.name)

            # Dropout
            output = dropout_layer(
                input_layer=output,
                drop_rate=self.hyper_params['drop_rate'],
                seed=self.seed,
                training=is_training,
                name='dropout0',
            )

            # Convolution
            output = conv_layer(
                input_layer=output,
                kernel_size=self.hyper_params['kernel_size'],
                strides=1,
                dilation_rate=1,
                filters=self.hyper_params['conv_filts'] * 2,
                padding='SAME',
                activation=tf.nn.relu,
                use_bias=False,
                name='conv1',
                seed=self.seed,
            )

            # Dropout
            output = dropout_layer(
                input_layer=output,
                drop_rate=self.hyper_params['drop_rate'],
                seed=self.seed,
                training=is_training,
                name='dropout1',
            )

            # Print shape
            print_output_shape(layer_name='output_conv1', net=output, print_shape=print_shape)

            # Convolution
            output = conv_layer(
                input_layer=output,
                kernel_size=self.hyper_params['kernel_size'],
                strides=1,
                dilation_rate=1,
                filters=self.hyper_params['conv_filts'] * 4,
                padding='SAME',
                activation=tf.nn.relu,
                use_bias=False,
                name='conv2',
                seed=self.seed,
            )

            # Dropout
            output = dropout_layer(
                input_layer=output,
                drop_rate=self.hyper_params['drop_rate'],
                seed=self.seed,
                training=is_training,
                name='dropout2',
            )

            # Print shape
            print_output_shape(layer_name='output_conv2', net=output, print_shape=print_shape)

            """Network Output"""
            # --- Global Average Pooling Layer ----------------------------------------------------------------------- #

            # Set name
            layer_name = 'gap'

            # Set layer scope
            with tf.variable_scope(layer_name):

                # Reduce mean along dimension 1
                gap = tf.reduce_mean(input_tensor=output, axis=1)

            # Print shape
            print_output_shape(layer_name=layer_name, net=gap, print_shape=print_shape)

            # --- Add Features --------------------------------------------------------------------------------------- #

            # # Set name
            # layer_name = 'add_features'
            #
            # # Set layer scope
            # with tf.variable_scope(layer_name):
            #
            #     # Add age and sex along dimension 1
            #     features = tf.concat(values=[gap, tf.cast(age, tf.float32), tf.cast(sex, tf.float32)], axis=1)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name, net=features, print_shape=print_shape)

            # --- Softmax Layer -------------------------------------------------------------------------------------- #

            # Set name
            layer_name = 'logits'

            # Softmax activation
            logits = fc_layer(
                input_layer=gap,
                neurons=self.classes,
                activation=None,
                use_bias=False,
                name=layer_name,
                seed=self.seed,
            )

            # Print shape
            print_output_shape(layer_name=layer_name, net=logits, print_shape=print_shape)

            # Compute Class Activation Maps
            cams = self._get_cams(net=output, is_training=is_training)

        return logits, cams

    def _residual_block(
        self,
        input_layer,
        kernel_size,
        layer_name,
        conv_filts,
        res_filts,
        skip_filts,
        dilation_rate,
        is_training,
        res=True,
        skip=True,
    ):
        """Wavenet residual block."""
        # Set layer scope
        with tf.variable_scope(layer_name):

            # Outputs dictionary
            outputs = dict()

            # Convolution tanh
            conv_filt = conv_layer(
                input_layer=input_layer,
                kernel_size=kernel_size,
                strides=1,
                dilation_rate=dilation_rate,
                filters=conv_filts,
                padding='SAME',
                activation=tf.nn.tanh,
                use_bias=False,
                name=layer_name + '_conv_filt',
                seed=self.seed,
            )

            # Convolution sigmoid
            conv_gate = conv_layer(
                input_layer=input_layer,
                kernel_size=kernel_size,
                strides=1,
                dilation_rate=dilation_rate,
                filters=conv_filts,
                padding='SAME',
                activation=tf.nn.sigmoid,
                use_bias=False,
                name=layer_name + '_conv_gate',
                seed=self.seed,
            )

            # Combine activations
            with tf.variable_scope('gate') as scope:
                activation = tf.multiply(conv_filt, conv_gate, name=scope.name)

            # Residual
            if res:
                # Convolution
                outputs['res'] = conv_layer(
                    input_layer=activation,
                    kernel_size=1,
                    strides=1,
                    dilation_rate=dilation_rate,
                    filters=res_filts,
                    padding='SAME',
                    activation=None,
                    use_bias=False,
                    name=layer_name + '_conv_res',
                    seed=self.seed,
                )

                # Add identity
                outputs['res'] = tf.add(outputs['res'], input_layer, name=layer_name + '_add_identity')

            # Skip
            if skip:
                # Convolution
                outputs['skip'] = conv_layer(
                    input_layer=activation,
                    kernel_size=1,
                    strides=1,
                    dilation_rate=dilation_rate,
                    filters=skip_filts,
                    padding='SAME',
                    activation=None,
                    use_bias=False,
                    name=layer_name + '_conv_skip',
                    seed=self.seed,
                )

        return outputs

    def _get_cams(self, net, is_training):
        """Collect class activation maps (CAMs)."""
        # Empty list for class activation maps
        cams = list()

        # Compute class activation map
        # with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
        if is_training is not None:
            for label in range(self.classes):
                cams.append(self._compute_cam(net=net, label=label))

        return tf.concat(cams, axis=2)

    def _compute_cam(self, net, label):
        """Compute class activation map (CAM) for specified label."""
        # Compute logits weights
        weights = self._get_logit_weights(net=net, label=label)

        # Compute class activation map
        cam = tf.matmul(net, weights)

        return cam

    def _get_logit_weights(self, net, label):
        """Get logits weights for specified label."""
        # Get number of filters in the final output
        num_filters = int(net.shape[-1])

        with tf.variable_scope('logits', reuse=True):
            weights = tf.gather(tf.transpose(tf.get_variable('kernel')), label)[0:num_filters]
            weights = tf.reshape(weights, [-1, num_filters, 1])

        # Reshape weights
        weights = self._reshape_logit_weights(net=net, weights=weights)

        return weights

    @staticmethod
    def _reshape_logit_weights(net, weights):
        """Reshape logits shapes to batch size for multiplication with net output."""
        return tf.tile(input=weights, multiples=[tf.shape(net)[0], 1, 1])

    def create_placeholders(self):
        """Creates place holders: waveform and label."""
        with tf.variable_scope('waveform') as scope:
            waveform = tf.placeholder(
                dtype=tf.float32, shape=[None, self.length, self.channels], name=scope.name
            )

        with tf.variable_scope('label') as scope:
            label = tf.placeholder(dtype=tf.int32, shape=[None, self.classes], name=scope.name)

        return waveform, label

    def create_generator(self, lookup_path, mode, batch_size):
        """Create data generator graph operation."""
<<<<<<< HEAD
        return DataGenerator(lookup_path=lookup_path, mode=mode, shape=[self.length, self.channels],
                             batch_size=batch_size, fs=self.hyper_params['fs'], prefetch_buffer=200,
                             seed=0, num_parallel_calls=32)
=======
        return DataGenerator(
            data_path=data_path,
            lookup_path=lookup_path,
            mode=mode,
            shape=[self.length, self.channels],
            batch_size=batch_size,
            prefetch_buffer=200,
            seed=0,
            num_parallel_calls=32,
        )
>>>>>>> DS

    def compute_metrics(self, logits, labels):
        """Computes the model accuracy for set of logits and labels."""
        with tf.variable_scope('metrics'):

<<<<<<< HEAD
=======
            # Get Sigmoid
            sigmoid = tf.nn.sigmoid(logits)

            # Apply Normal Rhythm correction
            sigmoid = tf.reshape(
                tf.py_func(func=self._normal_rhythm_correction, inp=[sigmoid], Tout=[tf.float32]),
                shape=[-1, self.classes],
            )

>>>>>>> DS
            # Get prediction
            predictions = tf.cast(tf.math.round(tf.nn.sigmoid(logits)), tf.int32)

            # Get label
            labels = tf.cast(labels, tf.int32)

<<<<<<< HEAD
            # Compute accuracy
            accuracy = tf.py_func(func=compute_accuracy, inp=[labels, predictions], Tout=tf.float64)

            # Compute F-measures
            macro_f_measure = tf.py_func(func=compute_f_measure, inp=[labels, predictions], Tout=tf.float64)

            # Compute Beta-measures
            macro_f_beta_measure, macro_g_beta_measure = tf.py_func(func=compute_beta_measures,
                                                                    inp=[labels, predictions, 2],
                                                                    Tout=[tf.float64, tf.float64])

            # Compute AUC
            macro_auroc, macro_auprc = tf.py_func(func=compute_auc, inp=[labels, predictions],
                                                  Tout=[tf.float64, tf.float64])

            # Compute challenge metric
            challenge_metric = tf.py_func(func=Metric().compute, inp=[labels, predictions], Tout=tf.float64)

            return (accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure,
                    macro_auroc, macro_auprc, challenge_metric)

    # def compute_metrics(self, logits, labels):
    #     """Computes the model accuracy for set of logits and labels."""
    #     with tf.variable_scope('metrics'):
    #
    #         # Get Sigmoid
    #         sigmoid = tf.nn.sigmoid(logits)
    #
    #         # Apply Normal Rhythm correction
    #         sigmoid = tf.reshape(tf.py_func(func=self._normal_rhythm_correction, inp=[sigmoid], Tout=[tf.float32]),
    #                              shape=[-1, self.classes])
    #
    #         # Get prediction
    #         predictions = tf.cast(tf.math.round(sigmoid), tf.int32)
    #
    #         # Get label
    #         labels = tf.cast(labels, tf.int32)
    #
    #         # Get metrics
    #         _, _, f_beta, g_beta = tf.py_func(func=compute_beta_score,
    #                                           inp=[labels, predictions, 2, self.classes, False],
    #                                           Tout=[tf.float64, tf.float64, tf.float64, tf.float64])
    #
    #         return f_beta, g_beta, tf.py_func(func=gmean, inp=[[f_beta, g_beta]], Tout=[tf.float64])
    #
    # @staticmethod
    # def _normal_rhythm_correction(sigmoid):
    #     for index in range(sigmoid.shape[0]):
    #         if sigmoid[index, 3] >= 0.75 and np.argmax(sigmoid[index, :]) == 3:
    #             sigmoid[index, [0, 1, 2, 4, 5, 6, 7, 8]] = 0.
    #     return sigmoid
=======
            # Get metrics
            _, _, f_beta, g_beta = tf.py_func(
                func=compute_beta_score,
                inp=[labels, predictions, 2, self.classes, False],
                Tout=[tf.float64, tf.float64, tf.float64, tf.float64],
            )

            return f_beta, g_beta, tf.py_func(func=gmean, inp=[[f_beta, g_beta]], Tout=[tf.float64])

    @staticmethod
    def _normal_rhythm_correction(sigmoid):
        for index in range(sigmoid.shape[0]):
            if sigmoid[index, 3] >= 0.75 and np.argmax(sigmoid[index, :]) == 3:
                sigmoid[index, [0, 1, 2, 4, 5, 6, 7, 8]] = 0.0
        return sigmoid
>>>>>>> DS
