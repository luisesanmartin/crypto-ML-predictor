import sys
import pickle
import pandas as pd
from sklearn.model_selection import ParameterGrid

sys.path.insert(1, '../utils')
import ml_utils as mlu

months = [
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
    Y = Y['label']

    for model in mlu.MODELS:

        parameters_list = list(ParameterGrid(mlu.PARAMETERS[model]))

        for i, parameters in enumerate(parameters_list):

            print('Model', model+str(i),'for', month)

            clf = mlu.MODELS[model](**parameters)
            clf = clf.fit(X, Y)

            file = export_path + month + '_' + model + str(i) + '.pickle'

            with open(file, 'wb') as f:
                pickle.dump(clf, f)
