import sys
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(1, '../utils')
import ml_utils as mlu
import feature_engineering_utils as feu

months = [
    'sep2021',
    'aug2021'
]
x_path = '../../data/working/test/X/'
y_path = '../../data/working/test/Y/'
classifiers_path = '../../classifiers/'
predictions_path = '../../data/working/test/predictions/'

for month in months:

    X = pd.read_csv(x_path+month+'_X.csv')
    Y = pd.read_csv(y_path+month+'_Y.csv')
    time_col = X['time']
    X = X.drop('time', 1)
    Y = Y['label']

    for file in os.listdir(classifiers_path):

        if file.startswith(month):

            with open(classifiers_path + file, 'rb') as f:
                model = pickle.load(f)
            model_name = file.split('_')[1].split('.')[0]

            Y_pred = model.predict(X)
            Y_pred_df = pd.DataFrame()
            Y_pred_df['time'] = time_col
            Y_pred_df['predicted'] = Y_pred
            Y_pred_df['actual'] = Y

            accuracy = accuracy_score(Y, Y_pred)

            print('Model'+model_name+'for'+month+'has an accuracy of '+str(round(accuracy*100,1))+'%')

            export_file = predictions_path + month + '_' + model_name + '_predictions.csv'
            Y_pred_df.to_csv(export_file, index=False)
