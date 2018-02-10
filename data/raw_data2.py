import pandas as pd

data_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
def load_csv(f_name, dtype=None, parse_dates=[]):
    return pd.read_csv(data_dir + f_name + '.csv', dtype=dtype, parse_dates=parse_dates)

air_reserve = load_csv('air_reserve', {
                                        'air_store_id': 'category',
                                        'visit_datetime': str,
                                        'reserve_datetime': str,
                                        'reserve_visitors': int},
                       ['visit_datetime', 'reserve_datetime']
                      )
hpg_reserve = load_csv('hpg_reserve', {
                                        'hpg_store_id': 'category',
                                        'visit_datetime': str,
                                        'reserve_datetime': str,
                                        'reserve_visitors': int},
                       ['visit_datetime', 'reserve_datetime']
                      )
air_store_info = load_csv('air_store_info', {
    'air_store_id': 'category',
    'air_genre_name': 'category',
    'air_area_name': 'category',
    'latitude': float,
    'longitude': float
})
hpg_store_info = load_csv('hpg_store_info', {
    'hpg_store_id': 'category',
    'hpg_genre_name': 'category',
    'hpg_area_name': 'category',
    'latitude': float,
    'longitude': float
})
store_id_relation = load_csv('store_id_relation', {
    'hpg_store_id': 'category',
    'air_store_id': 'category'
})
air_visit_data = load_csv('air_visit_data', {
    'air_store_id': 'category',
    'visit_date': str,
    'visitors': int
}, ['visit_date'])
date_info = load_csv('date_info', {
    'calendar_date': str,
    'day_of_week': 'category',
    'holiday_flg': bool
}, ['calendar_date']).rename(columns={'calendar_date': 'visit_date'})

sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')