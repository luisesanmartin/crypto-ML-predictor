import sys
import pickle

sys.path.insert(1, '../utils')
import data_fetching_utils as dfu

start = '2021-03-01T00:00:00'
end = '2021-05-01T01:00:00'
crypto = 'BTC'
freq = 10 # in minutes

data = dfu.get_data(crypto=crypto,
                    period=str(freq) + 'MIN',
                    start=start,
                    end=end)

file = '../../data/raw/data_' + crypto + '_' + str(freq) + \
       'min_' + start[:10] + '_' + end[:10] + '.txt'

print('\nSaving data...')
with open(file, 'wb') as f:
    pickle.dump(data, f)

print('\nData saved in', file)
