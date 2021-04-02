import sys
import pickle

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu

dates = {
    'feb2021': '2021-02-01T00:00:00',
    'jan2021': '2021-01-01T00:00:00',
    'dec2020': '2020-12-01T00:00:00',
    'nov2020': '2020-11-01T00:00:00',
    'oct2020': '2020-10-01T00:00:00',
    'sep2020': '2020-09-01T00:00:00',
    'aug2020': '2020-08-01T00:00:00',
    'jul2020': '2020-07-01T00:00:00',
    'jun2020': '2020-06-01T00:00:00',
    'may2020': '2020-05-01T00:00:00'
}
time_range_obs = 30 # in days
time_range_train = 60 # in minutes
freq = 10 # granularity of obs: 10 minutes

data_file = '../../data/raw/maxdata_BTC_10min_2021-03-17.txt'
with open(data_file, 'rb') as f:
    data = pickle.load(f)
data_dic = feu.transform_data_to_dict(data)

for month, date in dates.items():
    print('\nArranging data for', month)

    # Training X
    df_X, df_Y = feu.initial_train_X_brute_force(
        data_dic,
        date,
        time_range_obs,
        time_range_train,
        freq
        )
    export_file_X = '../../data/working/brute-force-approach/initial-train/' + month + '_X.csv'
    export_file_Y = '../../data/working/brute-force-approach/initial-train/' + month + '_y.csv'
    df_X.to_csv(export_file_X, index=False)
    df_Y.to_csv(export_file_Y, index=False)

    # Training labels
