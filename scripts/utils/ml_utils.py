from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_xgboost(X_train, Y_train):

    '''
    '''

    model = XGBClassifier(verbosity = 0)
    model.fit(X_train, Y_train)

    return model

def get_prediction(score, threshold=0.5):

    '''
    '''

    if score > threshold:
        return 1

    else:
        return 0
