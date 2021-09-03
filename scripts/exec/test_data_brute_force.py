import sys
import pickle

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu

dates = {
    'aug2021': '2021-08-01T00:00:00',
    'jul2021': '2021-07-01T00:00:00',
    'jun2021': '2021-06-01T00:00:00',
    'may2021': '2021-05-01T00:00:00',
    'apr2021': '2021-04-01T00:00:00'
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

    # Loading means and sds
    mean_sd_file = '../../data/working/train/X/' + month + 'mean_sd.txt'
    with open(mean_sd_file, 'rb') as f:
        mean_sd_list = pickle.load(f)

    # Test data
    df_X, df_Y = feu.test_set_brute_force(
        data_dic,
        date,
        time_range_obs,
        time_range_test,
        obs_freq,
        prediction_freq
        )
    df_X_sd = feu.standardize_df(df_X, stats=mean_sd_list)

    df_X_sd, df_Y = feu.match_dates(df_X_sd, df_Y)

    export_file_X_sd = '../../data/working/test/X/' + month + '_X.csv'
    export_file_Y = '../../data/working/test/Y/' + month + '_Y.csv'
    df_X_sd.to_csv(export_file_X_sd, index=False)
    df_Y.to_csv(export_file_Y, index=False)
