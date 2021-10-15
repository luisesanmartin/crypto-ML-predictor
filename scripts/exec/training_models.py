import sys
import pickle
import pandas as pd

sys.path.insert(1, '../utils')
import ml_utils as ml

months = [
    'oct2021',
    'sep2021',
    'aug2021'
]
x_path = '../../data/working/train/X/'
y_path = '../../data/working/train/Y/'
export_path = '../../classifiers/'

for month in months:

    X = pd.read_csv(x_path+month+'_X.csv')
    X = X.drop('time', 1)
    Y = pd.read_csv(y_path+month+'_Y.csv')
    Y = Y.drop('time', 1)

    model = ml.train_xgboost(X, Y)
    file = export_path + month + 'xgb.txt'

    with open(file, 'wb') as f:
        pickle.dump(model, f)

    print('XGB model for', month, 'finished')
