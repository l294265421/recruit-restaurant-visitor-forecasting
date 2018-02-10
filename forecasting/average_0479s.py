'''
https://www.kaggle.com/nitinsurya/surprise-me-2-neural-networks-keras/output
https://www.kaggle.com/tejasrinivas/surprise-me-4-lb-0-479/output
https://www.kaggle.com/aharless/exclude-same-wk-res-from-nitin-s-surpriseme2-w-nn/output
https://www.kaggle.com/meli19/surprise-me-h2o-automl-version-ver5-lb-0-479/code
上面四个kernel都声称自己的结果是0.479，我拿来没做任何修改就提交，结果都是0.480
对这4个结果进行平均，结果是0.478
'''
from data.raw_data import data_dir
import os
import pandas as pd
import numpy as np

base_dir = data_dir + '0.479s\\'

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
merge_dfs[['id', 'visitors']].to_csv(base_dir + 'average_0.479.csv', index=False)


