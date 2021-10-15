from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# list of classifiers
MODELS = {
    'Random forest': RandomForestClassifier,
    'Linear SVC': LinearSVC,
    'Logistic regression': LogisticRegression,
    'Gaussian NB': GaussianNB
}

# Parameters
PARAMETERS = {
    'Random forest': {'n_estimators': [100, 1000, 10000],
                      'criterion': ['gini', 'entropy'],
                      'max_features': [0.1, 0.2, 1/3, 1/2],
                      'n_jobs': [10],
                      'random_state': [793402]},
    'Linear SVC': {'C': [0.001, 0.01, 0.1, 1, 10],
                   'penalty': ['l1', 'l2'],
                   'dual': [False],
                   'random_state': [793402]},
    'Logistic regression': {'C': [0.001, 0.01, 0.1, 1, 10],
                            'penalty': ['l1', 'l2'],
                            'random_state': [793402]},
    'GaussianNB': {'priors': None}
}

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
