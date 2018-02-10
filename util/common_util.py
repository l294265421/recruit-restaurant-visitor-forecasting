import pandas as pd
import datetime
from datetime import timedelta

def wc(file_path):
    count = -1
    with open(file_path, 'rU') as f:
        for count, line in enumerate(open(file_path, 'rU')):
            pass
    count += 1
    return count

def head(file_path, n):
    return pd.read_csv(file_path, iterator=True).get_chunk(n)

def sample_submission_id(file_path):
    return [i for i in pd.read_csv(file_path, skip_blank_lines=True, header=0)['id']]

def unique_value(file_path, line_index):
    s = set()
    with open(file_path) as f:
        f.readline()
        for line in f:
            if line.strip():
                elements = line.split(',')
                s.add(elements[line_index])
    return list(s)


def get_days_after_today(dt, n=0):
    n_days_after = dt + timedelta(days=n)
    return datetime.datetime(n_days_after.year, n_days_after.month, n_days_after.day, n_days_after.hour, n_days_after.minute, n_days_after.second)

def merge_submit(base_dir, prefix, suffix):
    weeks = [16, 17, 18, 19, 20, 21, 22]
    submit_part = []
    for week in weeks:
        submit_part.append(pd.read_csv(base_dir + prefix + '_' + str(week) + suffix))
    pd.concat(submit_part, axis=0).to_csv(base_dir + prefix + suffix, index=False)

def split_submit(base_dir, submit_file_name):
    submit_file_path = base_dir + submit_file_name
    submit = pd.read_csv(submit_file_path)
    submit['weekofyear'] = pd.to_datetime(submit['id'].map(lambda x: str(x).split('_')[2])).dt.weekofyear

    for i in range(16, 23):
        last_dot_index = submit_file_path.rfind('.')
        submit_part_path = submit_file_path[:last_dot_index]
        submit_part_path += '_'
        submit_part_path += str(i)
        submit_part_path += submit_file_path[last_dot_index:]
        submit_part_data = submit[submit['weekofyear'] == i]
        submit_part_data[['id', 'visitors']].to_csv(submit_part_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Recruit Restaurant Visitor Forecasting\\'
    merge_submit(base_dir + 'for_merge_test\\', 'submission_use_stacking3', '.csv')
    # split_submit(base_dir, 'best_submission.csv')
    # train_file = base_dir + 'train.csv'
    # test_file = base_dir + 'test.csv'
    # sampleSubmission = base_dir + 'sampleSubmission.csv'
    # print(wc(base_dir + 'train.csv'))
    # print(wc(base_dir + 'test.csv'))
    # print(wc(r'D:\Users\liyuncong\PycharmProjects\avazu-click-through-rate-prediction\util\common_util.py'))
    # train_head_n = head(train_file, 50000)
    # train_head_n.to_csv(base_dir + 'train_head.csv', index=False)
    # test_head_n = head(test_file, 5000)
    # test_head_n.to_csv(base_dir + 'test_head.csv', index=False)
    # sampleSubmission_head = head(sampleSubmission, 5000)
    # sampleSubmission_head.to_csv(base_dir + 'sampleSubmission_head.csv', index=False)
    # wide_result_head = head(base_dir + 'va.r0.site.sp', 5)
    # print(wide_result_head)
    # wide_result_head.to_csv(base_dir + 'tr.r0.app.new.csv_head.csv', index=False)
    # count = 0
    # with open(train_file) as t:
    #     for line in t:
    #         parts = line.split(',')
    #         if '1000009418151094273' == parts[0]:
    #             count += 1

    # print(count)
    # unique_C1 = unique_value(base_dir + 'train_head.csv', 3)
    # for element in unique_C1:
    #     print(element)