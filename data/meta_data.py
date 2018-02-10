import pandas as pd
from data.raw_data2 import data_dir
from data.features3 import categorical_columns

train_file_path = data_dir + 'train.csv'
test_file_path = data_dir + 'test.csv'

train_data = pd.read_csv(train_file_path, nrows=10)
test_data = pd.read_csv(test_file_path, nrows=10)

train_columns = list(train_data.columns)
print('train_columns')
print(train_columns)

test_columns = list(test_data.columns)
print('')
print(test_columns)
with open(data_dir + 'test_fields.csv', 'w') as test_fields_file:
    for field in test_columns:
        test_fields_file.write("'" + field + "',\n")

test_columns_not_in_train_columns = [column for column in test_columns if column not in train_columns]
print('test_columns_not_in_train_columns')
print(test_columns_not_in_train_columns)

train_columns_not_in_test_columns = [column for column in train_columns if column not in test_columns]
print('train_columns_not_in_test_columns')
print(train_columns_not_in_test_columns)

need_normalized_numerical_columns = [column for column in test_columns if column]

dont_need_normalized_numerical_columns =[]