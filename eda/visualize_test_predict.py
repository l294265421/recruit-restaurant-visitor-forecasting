from data.raw_data import data_dir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy.random import randn
import os

def my_plot(base_dir, air_store_id):
    dfs = []
    for file_path in os.listdir(base_dir):
        dfs.append(pd.read_csv(base_dir + '\\' + file_path))

    data = dfs[0]
    for i in range(len(dfs)):
        data[str(i)] = dfs[i]['visitors']

    data['visit_date'] = data['id'].map(lambda x: str(x).split('_')[2])
    data['air_store_id'] = data['id'].map(
        lambda x: '_'.join(x.split('_')[:2]))

    data = data[air_store_id == data['air_store_id']]

    data = data[['visit_date'] + [str(i) for i in range(len(dfs))]]

    data.plot(x='visit_date', y=[str(i) for i in range(len(dfs))])

    plt.show()

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\test_predict_dir\all'

my_plot(base_dir, 'air_0164b9927d20bcc3')

print('finish')
