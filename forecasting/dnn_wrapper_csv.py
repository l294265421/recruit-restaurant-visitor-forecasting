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

_CSV_COLUMNS = ['air_store_id',
 'visit_date',
 'visitors',
 'dow',
 'year',
 'month',
 'day_of_week',
 'holiday_flg',
 'min_visitors',
 'mean_visitors',
 'median_visitors',
 'max_visitors',
 'count_observations',
 'air_genre_name',
 'air_area_name',
 'latitude',
 'longitude',
 'air_genre_name0',
 'air_area_name0',
 'air_genre_name1',
 'air_area_name1',
 'air_genre_name2',
 'air_area_name2',
 'air_genre_name3',
 'air_area_name3',
 'air_genre_name4',
 'air_area_name4',
 'air_genre_name5',
 'air_area_name5',
 'air_genre_name6',
 'air_area_name6',
 'air_genre_name7',
 'air_area_name7',
 'air_genre_name8',
 'air_area_name8',
 'air_genre_name9',
 'air_area_name9',
 'rs1_x',
 'rv1_x',
 'rs2_x',
 'rv2_x',
 'rs1_y',
 'rv1_y',
 'rs2_y',
 'rv2_y',
 'id',
 'total_reserv_sum',
 'total_reserv_mean',
 'total_reserv_dt_diff_mean',
 'date_int',
 'var_max_lat',
 'var_max_long',
 'lon_plus_lat',
 'air_store_id2']

CATEGORICAL = [
    'dow',
    'year',
    'month',
    'day_of_week',
]

NUMEIRCAL = [
 'holiday_flg',
 'min_visitors',
 'mean_visitors',
 'median_visitors',
 'max_visitors',
 'count_observations',
 'air_genre_name',
 'air_area_name',
 'latitude',
 'longitude',
 'air_genre_name0',
 'air_area_name0',
 'air_genre_name1',
 'air_area_name1',
 'air_genre_name2',
 'air_area_name2',
 'air_genre_name3',
 'air_area_name3',
 'air_genre_name4',
 'air_area_name4',
 'air_genre_name5',
 'air_area_name5',
 'air_genre_name6',
 'air_area_name6',
 'air_genre_name7',
 'air_area_name7',
 'air_genre_name8',
 'air_area_name8',
 'air_genre_name9',
 'air_area_name9',
 'rs1_x',
 'rv1_x',
 'rs2_x',
 'rv2_x',
 'rs1_y',
 'rv1_y',
 'rs2_y',
 'rv2_y',

 'total_reserv_sum',
 'total_reserv_mean',
 'total_reserv_dt_diff_mean',
 'date_int',
 'var_max_lat',
 'var_max_long',
 'lon_plus_lat',
 'air_store_id2']

_CSV_COLUMN_DEFAULTS = [[''],
 [''],
 [0],
 [0],
 [0],
 [0],
 [0],
 [0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [0.0],
 [''],
 [0.0],
 [0.0],
 [0.0],
 [0],
 [0.0],
 [0.0],
 [0.0],
 [0]]

_NUM_EXAMPLES = {
    'train': 252108,
    'validation': 32019,
}

class DnnRegressor:
    def __init__(self, base_dir, train_filename, test_filename):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_dir', type=str, default='model/wide_deep',
            help='Base directory for the model.')
        parser.add_argument(
            '--model_type', type=str, default='deep',
            help="Valid model types: {'wide_deep', 'deep', 'wide_deep'}.")
        parser.add_argument(
            '--train_epochs', type=int, default=1, help='Number of training epochs.')
        parser.add_argument(
            '--epochs_per_eval', type=int, default=1,
            help='The number of training epochs to run between evaluations.')
        parser.add_argument(
            '--batch_size', type=int, default=512, help='Number of examples per batch.')
        parser.add_argument(
            '--train_data', type=str, default=base_dir+train_filename,
            help='Path to the training data.')

        parser.add_argument(
            '--test_data', type=str, default=base_dir+test_filename,
            help='Path to the test data.')

        tf.logging.set_verbosity(tf.logging.INFO)
        FLAGS, unparsed = parser.parse_known_args()
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
        self.FLAGS = FLAGS
        self.base_dir = base_dir
        self.train_filename = train_filename
        self.test_filename = test_filename

    def fit(self):
        def categorical_column_with_vocabulary_list(row_column_name):
            file_path = self.base_dir + self.train_filename
            return tf.feature_column.categorical_column_with_vocabulary_list(row_column_name, cu.unique_value(file_path,
                                                                                                              _CSV_COLUMNS.index(
                                                                                                                  row_column_name)))
        def unique_feature_num(row_column_name):
            file_path = self.base_dir + self.train_filename
            return len(cu.unique_value(file_path, _CSV_COLUMNS.index(row_column_name)))

        def build_model_columns():
            base_columns = []
            for column_name in NUMEIRCAL:
                base_columns.append(tf.feature_column.numeric_column(column_name))
            for column_name in CATEGORICAL:
                base_columns.append(categorical_column_with_vocabulary_list(column_name))

            wide_columns = base_columns

            deep_columns = base_columns

            return wide_columns, deep_columns

        def build_estimator(model_dir, model_type):
            """Build an estimator appropriate for the given model type."""
            wide_columns, deep_columns = build_model_columns()
            hidden_units = [100, 75, 50, 25]

            # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
            # trains faster than GPU for this model.
            run_config = tf.estimator.RunConfig().replace(
                session_config=tf.ConfigProto(device_count={'GPU': 0}))
            if model_type == 'wide_deep':
                return tf.estimator.LinearRegressor(
                    model_dir=model_dir,
                    feature_columns=wide_columns,
                    config=run_config)
            elif model_type == 'deep':
                return tf.estimator.DNNRegressor(
                    model_dir=model_dir,
                    feature_columns=deep_columns,
                    hidden_units=hidden_units,
                    config=run_config)
            else:
                return tf.estimator.DNNLinearCombinedRegressor(
                    model_dir=model_dir,
                    linear_feature_columns=wide_columns,
                    dnn_feature_columns=deep_columns,
                    dnn_hidden_units=hidden_units,
                    config=run_config)

        model = build_estimator(self.FLAGS.model_dir, self.FLAGS.model_type)

        model.train(input_fn=lambda: self.input_fn(self.base_dir + self.train_filename, self.FLAGS.epochs_per_eval, False, self.FLAGS.batch_size, False))
        self.model = model

    def predict(self):
        predictions = list(self.model.predict(input_fn=lambda: self.input_fn(self.base_dir + self.test_filename, 1, False, self.FLAGS.batch_size, True)))
        predicted = [float('%.4f' % (p["predictions"][0])) for p in predictions]
        return predicted

    def input_fn(self, data_file, num_epochs, shuffle, batch_size, predict):
        """Generate an input function for the Estimator."""
        assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have either run data_download.py or '
            'set both arguments --train_data and --test_data.' % data_file)

        def parse_csv(value):
            print('Parsing', data_file)
            columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
            features = dict(zip(_CSV_COLUMNS, columns))

            labels = features.pop('id')
            if not predict:
                labels = features.pop('visitors')
            return features, labels

        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(data_file)

        # Skip header row
        dataset = dataset.skip(1)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

        dataset = dataset.map(parse_csv, num_parallel_calls=5)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

if __name__ == '__main__':
    dw = DnnRegressor(data_dir, 'train.csv', 'test.csv')
    dw.fit()
    predicted = dw.predict()
    print(predicted)