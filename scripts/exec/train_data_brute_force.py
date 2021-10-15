import sys
import pickle

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu

dates = {
    'oct2021': '2021-10-01T00:00:00',
    'sep2021': '2021-09-01T00:00:00',
    'aug2021': '2021-08-01T00:00:00',
}

time_range_obs = 30 # in days
time_range_train = 60 # in minutes
obs_freq = 10 # granularity of obs
prediction_freq = 30 # frequency of predictions

data_file = '../../data/working/total_data.txt'
with open(data_file, 'rb') as f:
    data_dic = pickle.load(f)

for month, date in dates.items():
    print('\nArranging training data for', month, '(backward looking)')

    # Training data
    df_X, df_Y = feu.train_set_brute_force(
        data_dic,
        date,
        time_range_obs,
        time_range_train,
        obs_freq,
        prediction_freq
        )

    df_X, df_Y = feu.match_dates(df_X, df_Y)

    export_file_Y = '../../data/working/train/Y/' + month + '_Y.csv'
    export_file_X = '../../data/working/train/X/' + month + '_X.csv'
    df_X.to_csv(export_file_X, index=False)
    df_Y.to_csv(export_file_Y, index=False)
