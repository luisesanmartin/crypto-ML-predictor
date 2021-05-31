import sys
import pickle

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu

dates = {
    'may2021': '2021-05-01T00:00:00',
    'apr2021': '2021-04-01T00:00:00',
    'mar2021': '2021-03-01T00:00:00',
    'feb2021': '2021-02-01T00:00:00',
    'jan2021': '2021-01-01T00:00:00',
    'dec2020': '2020-12-01T00:00:00',
    'nov2020': '2020-11-01T00:00:00',
    'oct2020': '2020-10-01T00:00:00',
    'sep2020': '2020-09-01T00:00:00',
    'aug2020': '2020-08-01T00:00:00',
    'jul2020': '2020-07-01T00:00:00',
    'jun2020': '2020-06-01T00:00:00'
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
