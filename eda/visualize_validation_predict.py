from data.raw_data import data_dir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy.random import randn

def my_plot(file_path, air_store_id):
    data = pd.read_csv(file_path)
    data['visit_date'] = data['id'].map(lambda x: str(x).split('_')[2])
    data['air_store_id'] = data['id'].map(
        lambda x: '_'.join(x.split('_')[:2]))

    data = data[air_store_id == data['air_store_id']]

    data = data[['visit_date', 'visitors', 'visitors_predict']]

    data.plot(x='visit_date', y=['visitors', 'visitors_predict'])

    plt.show()

file_name = 'validation_random_forest_regressor_16.csv'

my_plot(data_dir + file_name, 'air_a271c9ba19e81d17')

print('finish')
