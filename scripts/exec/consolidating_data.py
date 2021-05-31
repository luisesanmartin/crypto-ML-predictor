import os
import pickle
import sys

sys.path.insert(1, '../utils')
import feature_engineering_utils as feu


data_path = '../../data/raw/'
total_data = {}

for data_file in os.listdir(data_path):

    if data_file.endswith('.txt'):

        with open(data_path+data_file, 'rb') as f:
            data = pickle.load(f)

        total_data[data_file] = data

consolidated_data = {}

for data in total_data.values():

    data_dic = feu.transform_data_to_dict(data)
    consolidated_data = {**consolidated_data, **data_dic}

print('Total observations:', len(consolidated_data))
output = '../../data/working/total_data.txt'

with open(output, 'wb') as f:
    pickle.dump(consolidated_data, f)

print('Data saved in', output)
