import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(1, '../utils')
import ml_utils as ml
import feature_engineering_utils as feu

months = [
    'apr2021',
    'mar2021',
    'feb2021',
    'jan2021',
    'dec2020',
    'nov2020',
    'oct2020',
    'sep2020',
    'aug2020',
    'jul2020',
    'jun2020'
]
x_path = '../../data/working/test/X/'
y_path = '../../data/working/test/Y/'
classifiers_path = '../../classifiers/'
predictions_path = '../../data/working/test/predictions/'

for month in months:

    X = pd.read_csv(x_path+month+'_X.csv')
    Y = pd.read_csv(y_path+month+'_Y.csv')

    file = classifiers_path + month + 'xgb.txt'
    time_col = X['time']
    X = X.drop('time', 1)
    Y = Y.drop('time', 1)

    with open(file, 'rb') as f:
        model = pickle.load(f)

    Y_pred = model.predict(X)
    Y_pred_df = pd.DataFrame()
    Y_pred_df['time'] = time_col
    Y_pred_df['predictions'] = Y_pred

    predictions = [ml.get_prediction(value) for value in Y_pred]
    accuracy = accuracy_score(Y, predictions)

    print('Model for', month, 'has an accuracy of', str(accuracy))

    export_file = predictions_path + month + '_predictions.csv'
    Y_pred_df.to_csv(export_file, index=False)
