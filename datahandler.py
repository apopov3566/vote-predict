import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import compose

def load_train(fname, n_validate, max_total = -1):
    X = np.loadtxt(fname, skiprows = 1, delimiter = ",")
    np.random.shuffle(X)

    data = X[:max_total, 5:-1]
    cols = get_categorical_cols(data, 100)
    data = get_categorical(data, cols)

    test_data = data[:n_validate]
    test_labels = X[:n_validate, -1]
    train_data = data[n_validate:]
    train_labels = X[n_validate:max_total, -1]

    return train_data, train_labels, test_data, test_labels

def load_predict(fname):
    return np.loadtxt(fname, skiprows = 1, delimiter = ",")

def classification_err(y, real_y):
    diff = 0
    for i in range(len(y)):
        if(y[i] != real_y[i]):
            diff += 1/len(y)

    return diff

def auc_err(y, real_y):
    return metrics.roc_auc_score(real_y, y)

def get_categorical_cols(data, threshold):
    categoricals = []
    for i in range(len(data.T)):
        if len(np.unique(data.T[i])) < threshold:
            categoricals.append(i)
    return categoricals

def get_categorical(data, categoricals):
    ct = compose.make_column_transformer((categoricals, preprocessing.OneHotEncoder(categories='auto')))
    x = ct.fit_transform(data).toarray()
    return x
