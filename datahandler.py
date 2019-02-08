import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import compose

def load_train(fname, n_validate, max_total = -1):
    '''loads, splits, and changes categorical data to categories for
    training data'''
    X = np.loadtxt(fname, skiprows = 1, delimiter = ",")
    np.random.shuffle(X)

    data = X[:max_total, 5:-1]
    cols = get_categorical_cols(data, 100)
    data = make_categorical(data, cols)
    data = make_regularized(data)


    test_data = data[:n_validate]
    test_labels = X[:n_validate, -1]
    train_data = data[n_validate:]
    train_labels = X[n_validate:max_total, -1]

    return train_data, train_labels, test_data, test_labels

def load_predict(fname, cols):
    '''loads data to predict on'''
    data = np.loadtxt(fname, skiprows = 1, delimiter = ",")
    data = data[:, 5:]
    data = make_categorical(data, cols)
    data = make_regularized(data)

    return data

def classification_err(y, real_y):
    """gets classification error given regression y and real y"""
    cat_y = np.floor(2 * y)
    diff = 0
    for i in range(len(cat_y)):
        if(cat_y[i] != real_y[i]):
            diff += 1/len(cat_y)

    return diff

def auc_err(y, real_y):
    """gets area under curve error given regression y and real y"""
    return metrics.roc_auc_score(real_y, y)

def get_categorical_cols(data, threshold):
    """gets columns which should be categorical data"""
    categoricals = []
    for i in range(len(data.T)):
        if len(np.unique(data.T[i])) < threshold:
            categoricals.append(i)
    return categoricals

def make_categorical(data, categoricals):
    """converts categorical data to one-hot encoded"""
    ct = compose.make_column_transformer((categoricals, preprocessing.OneHotEncoder(categories='auto')))
    x = ct.fit_transform(data).toarray()
    return x

def reg(data):
    """regularizes column"""
    if np.std(data) > 0:
        return (data - np.average(data))/(np.std(data))
    return data - np.average(data)

def make_regularized(data):
    """regularizes all data"""
    X = np.insert(np.apply_along_axis(reg,0,data), 0, 1, axis=1)
    return X
