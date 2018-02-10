import pandas as pd
import numpy as np

data_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
air_reserve = pd.read_csv(data_dir + 'air_reserve.csv')
air_visit_data = pd.read_csv(data_dir + "air_visit_data.csv")
air_store_info = pd.read_csv(data_dir + "air_store_info.csv")
hpg_reserve = pd.read_csv(data_dir + "hpg_reserve.csv")
hpg_store_info = pd.read_csv(data_dir + "hpg_store_info.csv")
store_id_relation = pd.read_csv(data_dir + "store_id_relation.csv")
date_info = pd.read_csv(data_dir + "date_info.csv")
sample_submission = pd.read_csv(data_dir + "sample_submission.csv")

def unique_values(df, field_name):
    return df[field_name].unique()

def unique_value_num(df, field_name):
    return str(len(unique_values(df, field_name)))

def sample_submission_air_store_id():
    return sample_submission['id'].map(lambda e: '_'.join(e.split('_')[:2]))

def sample_submission_air_store_id_num():
    return str(len(sample_submission['id'].map(lambda e: '_'.join(e.split('_')[:2])).unique()))

def list1_contain_all_element_in_list2(list1, list2):
    for element in list2:
        if element not in list1:
            return False
    return True

def test_date_weekofyear():
    temp = sample_submission;
    temp['visit_date'] = temp['id'].map(lambda x: str(x).split('_')[2])
    temp['weekofyear'] = pd.to_datetime(temp['visit_date']).dt.weekofyear
    print(temp[['visit_date', 'weekofyear']])
    print('visit_date len:' + str(len(temp['visit_date'].unique())))

print("air_reserve中air_store_id数目：" + unique_value_num(air_reserve, 'air_store_id'))
print("air_visit_data中air_store_id数目：" + unique_value_num(air_visit_data, 'air_store_id'))
print("air_store_info中air_store_id数目：" + unique_value_num(air_store_info, 'air_store_id'))
print("store_id_relation中air_store_id数目：" + unique_value_num(store_id_relation, 'air_store_id'))
print("sample_submission中air_store_id数目：" + sample_submission_air_store_id_num())
print("sample_submission中air_store_id都在air_store_info中："
      + str(list1_contain_all_element_in_list2(unique_values(air_store_info, 'air_store_id'), sample_submission_air_store_id())))

print("hpg_reserve中hpg_store_id数目:" + unique_value_num(hpg_reserve, 'hpg_store_id'))
print("hpg_store_info中hpg_store_id数目:" + unique_value_num(hpg_store_info, 'hpg_store_id') )
print("store_id_relation中hpg_store_id数目:" + unique_value_num(store_id_relation, 'hpg_store_id'))
print(test_date_weekofyear())

