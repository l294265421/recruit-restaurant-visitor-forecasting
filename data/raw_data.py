import pandas as pd

data_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
data = {
    'air_visit_data': pd.read_csv(data_dir + 'air_visit_data.csv'),
    'air_store_info': pd.read_csv(data_dir + 'air_store_info.csv'),
    'hpg_store_info': pd.read_csv(data_dir + 'hpg_store_info.csv'),
    'air_reserve': pd.read_csv(data_dir + 'air_reserve.csv'),
    'hpg_reserve': pd.read_csv(data_dir + 'hpg_reserve.csv'),
    'store_id_relation': pd.read_csv(data_dir + 'store_id_relation.csv'),
    'sample_submission': pd.read_csv(data_dir + 'sample_submission.csv'),
    'date_info': pd.read_csv(data_dir + 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}