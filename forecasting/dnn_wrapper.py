from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import util.common_util as cu
from data.raw_data import data_dir

import tensorflow as tf
import pandas as pd

import math

class DnnRegressor:
    def __init__(self, numerical_columns, categorical_columns, mode_type='wide', train_epochs=2, batch_size=8):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_dir', type=str, default='model/wide_deep',
            help='Base directory for the model.')
        parser.add_argument(
            '--model_type', type=str, default=mode_type,
            help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
        parser.add_argument(
            '--train_epochs', type=int, default=train_epochs, help='Number of training epochs.')
        parser.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='The number of training epochs to run between evaluations.')
        parser.add_argument(
            '--batch_size', type=int, default=batch_size, help='Number of examples per batch.')

        tf.logging.set_verbosity(tf.logging.INFO)
        FLAGS, unparsed = parser.parse_known_args()
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
        self.FLAGS = FLAGS

    def fit(self, x, y):
        tf.logging.set_verbosity(tf.logging.INFO)
        def build_model_columns():
            base_numeric_columns = []
            for column_name in self.numerical_columns:
                base_numeric_columns.append(tf.feature_column.numeric_column(key=column_name))
            base_categorical_columns = []
            for column_name in self.categorical_columns:
                base_categorical_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(column_name, x[column_name].unique()))
            wide_columns = base_numeric_columns + base_categorical_columns

            deep_columns = base_numeric_columns
            for base_categorical_column in base_categorical_columns:
                deep_columns.append(tf.feature_column.indicator_column(base_categorical_column))

            return wide_columns, deep_columns

        def build_estimator(model_dir, model_type):
            """Build an estimator appropriate for the given model type."""
            wide_columns, deep_columns = build_model_columns()
            hidden_units = [128, 64, 32]

            # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
            # trains faster than GPU for this model.
            run_config = tf.estimator.RunConfig().replace(
                session_config=tf.ConfigProto(device_count={'GPU': 0}))
            if model_type == 'wide':
                return tf.estimator.LinearRegressor(
                    model_dir=model_dir,
                    feature_columns=wide_columns,
                    config=run_config)
            elif model_type == 'deep':
                return tf.estimator.DNNRegressor(
                    model_dir=model_dir,
                    feature_columns=deep_columns,
                    hidden_units=hidden_units,
                    dropout=0.1,
                    config=run_config)
            else:
                return tf.estimator.DNNLinearCombinedRegressor(
                    model_dir=model_dir,
                    linear_feature_columns=wide_columns,
                    dnn_feature_columns=deep_columns,
                    dnn_hidden_units=hidden_units,
                    config=run_config)

        shutil.rmtree(self.FLAGS.model_dir, ignore_errors=True)
        model = build_estimator(self.FLAGS.model_dir, self.FLAGS.model_type)

        model.train(input_fn=lambda: self.train_input_fn(x, y, self.FLAGS.batch_size))
        self.model = model

    def predict(self, x):
        predictions = list(self.model.predict(input_fn=lambda: self.eval_input_fn(x, None, self.FLAGS.batch_size)))
        predicted = [float('%.4f' % (p["predictions"][0])) for p in predictions]
        return predicted

    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        temp = dict(features)
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(len(features)).repeat(self.FLAGS.train_epochs).batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()

    def eval_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()
