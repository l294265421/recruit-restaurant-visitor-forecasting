'''
https://www.kaggle.com/meli19/ensemble-public-submissions
me
对这2个结果进行平均，结果是0.478
'''
from data.raw_data import data_dir
import os
import pandas as pd
import numpy as np

base_dir = data_dir + '0.478s\\'

file_paths = os.listdir(base_dir)

dfs = []
i = 0
for file_path in file_paths:
    dfs.append(pd.read_csv(base_dir + file_path).rename(columns={'visitors': 'visitors' + str(i)}))
    i += 1

merge_dfs = dfs[0]
for i in  range(1, len(dfs)):
    merge_dfs = pd.merge(merge_dfs, dfs[i], how='inner', on='id')

visitors_columns = ['visiters' + str(i)  for  i in range(len(dfs))]

merge_dfs['visitors'] = merge_dfs.apply(lambda x :(x.visitors0 + x.visitors1 + x.visitors2 + x.visitors3)/4, axis=1)
merge_dfs[['id', 'visitors']].to_csv(base_dir + 'average_0.478.csv', index=False)


