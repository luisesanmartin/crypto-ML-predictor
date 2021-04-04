import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(1, '../utils')
import ml_utils as ml

months = [
    'feb2021',
    'jan2021',
    'dec2020',
    'nov2020',
    'oct2020',
    'sep2020',
    'aug2020',
    'jul2020',
    'jun2020',
    'may2020'
]
x_path = '../../data/working/test/X/brute-force/'
classifiers_path = '../../classifiers/brute_force/'

for month in months:

    X = pd.read_csv(x_path+month+'_X.csv')
    X = X.drop('time', 1)
    file = classifiers_path + month + 'xgb.txt'

    with open(file, 'rb') as f:
        model = pickle.load(f)

    Y = model.predict(X)
    predictions = [round(value) for value in Y]
    accuracy = accuracy_score(Y, predictions)

    print('Model for', month, 'has an accuracy of', str(accuracy))
