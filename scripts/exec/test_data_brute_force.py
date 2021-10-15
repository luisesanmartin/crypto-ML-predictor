import sys
import pickle

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu

dates = {
    'sep2021': '2021-09-01T00:00:00',
    'aug2021': '2021-08-01T00:00:00'
}

time_range_obs = 30 # in days
time_range_test = 60 # in minutes
obs_freq = 10 # granularity of obs: 10 minutes
prediction_freq = 30 # in minutes

data_file = '../../data/working/total_data.txt'
#data_file = '../../data/raw/data_BTC_10min_2021-03-01_2021-05-01.txt'
with open(data_file, 'rb') as f:
    data_dic = pickle.load(f)

for month, date in dates.items():
    print('\nArranging testing data for', month, '(forward looking)')

    # Test data
    df_X, df_Y = feu.test_set_brute_force(
        data_dic,
        date,
        time_range_obs,
        time_range_test,
        obs_freq,
        prediction_freq
        )

    df_X, df_Y = feu.match_dates(df_X, df_Y)

    export_file_X = '../../data/working/test/X/' + month + '_X.csv'
    export_file_Y = '../../data/working/test/Y/' + month + '_Y.csv'
    df_X.to_csv(export_file_X, index=False)
    df_Y.to_csv(export_file_Y, index=False)
