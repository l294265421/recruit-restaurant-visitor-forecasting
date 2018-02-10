from data.raw_data import data_dir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy.random import randn
# from data.transformed_data2 import train_predict_dir_prefix

train_predict_dir_prefix = data_dir + 'train_predict_dir\\'

def my_plot(file_path, air_store_id):
    data = pd.read_csv(file_path, usecols=['air_store_id', 'visit_date', 'visitors'])

    data = data[air_store_id == data['air_store_id']]

    data = data[['visit_date', 'visitors']]

    data.plot(x='visit_date', y=['visitors'])

    plt.show()

def my_plot_day_of_week(file_path, air_store_id):
    data = pd.read_csv(file_path, usecols=['air_store_id', 'visit_date', 'visitors', 'day_of_week'])

    data = data[air_store_id == data['air_store_id']]

    dfs = []
    for i in range(7):
        df = data[data['day_of_week'] == i].sort_values(by='visit_date', axis=0, ascending=True).reset_index()
        dfs.append(df)

    data = dfs[0]
    for i in range(7):
        data[str(i)] = dfs[i]['visitors']
    data = data.fillna(0)
    data.plot(y=[str(i) for i in range(7)])

    plt.show()

file_name = 'train.csv'

my_plot_day_of_week(data_dir + '\\' + file_name, 8)

print('finish')
