import pandas as pd
import math

data_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
origion = pd.read_csv(data_dir + 'StackingSubmission.csv')
origion['visitors'] = origion['visitors'].clip(lower=0.)
origion['visitors'] = origion['visitors'].apply(lambda x : math.floor(x))
origion.to_csv(data_dir + 'StackingSubmission_t.csv', index=False)