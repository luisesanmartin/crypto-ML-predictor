import sys
import pickle

sys.path.insert(1, '../utils')
import data_fetching_utils

end = '2021-03-17T18:20:01'

data = data_fetching_utils.get_data_max_possible(end=end)

file = '../../data/raw/maxdata_BTC_10min_' + end[:10] + '.txt'

print('\nSaving data...')
with open(file, 'wb') as f:
    pickle.dump(data, f)

print('\nData saved in', file)
