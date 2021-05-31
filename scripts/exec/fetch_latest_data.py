from datetime import datetime
import sys
import pickle

sys.path.insert(1, '../utils')
import data_fetching_utils as dfu
import feature_engineering_utils as feu

data_dic_path = '../../data/working/total_data.txt'
with open(data_dic_path, 'rb') as f:
    data_dic = pickle.load(f)

now = datetime.now()
end = dfu.time_in_string(now)
start = feu.latest_time(data_dic)
freq = 10 # in minutes
crypto = 'BTC'

latest_data = dfu.get_data(crypto=crypto,
               period=str(freq) + 'MIN',
               start=start,
               end=end)

file = '../../data/raw/data_' + crypto + '_' + str(freq) + \
       'min_' + start[:10] + '_' + end[:10] + '.txt'

print('\nSaving data...')
with open(file, 'wb') as f:
    pickle.dump(latest_data, f)

print('\nData saved in', file)
