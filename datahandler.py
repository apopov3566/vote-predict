import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import compose

def load_train(fname, n_validate, max_total = -1):
    '''loads, splits, and changes categorical data to categories for
    training data'''
    X = np.loadtxt(fname, skiprows = 1, delimiter = ",")
    np.random.shuffle(X)

    data = X[:, 5:-1]
    cols = get_categorical_cols(data, 100)
    data = make_categorical(data, cols)
    data = make_regularized(data)
    data = data[:max_total]

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

def save_model(model, fname):
    """saves given model to file"""
    pickle.dump( model, open( fname, "wb" ) )

def load_model(fname):
    f = open(fname, 'rb')
    model = pickle.load(f)
    f.close()
    return model

def save_prediction(predict_results, fname):
    f = open(fname)
    f.write("id,target,\n")
    for i in range(len(predict_results)):
        f.write(str(i) + "," + str(predict_results[i]))
    f.close()

    
   

def collect_model_stats(data, labels, v_data, 
        v_labels, clf, clf_descriptor, verbose = True):
    
    train_result = clf.predict(data)
    validate_result = clf.predict(v_data)
    cat_result = np.floor(2 * validate_result)
    data_output = []
    data_output.append(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    data_output.append("Classification Error: " + str(classification_err(cat_result, v_labels)))
    data_output.append("AUC validation: " + str(auc_err(validate_result, v_labels)))
    data_output.append("AUC train: " + str(auc_err(train_result, labels)))

    with open(clf_descriptor+".dat", 'w') as out:
        for i in data_output:
            out.write(i)
            out.write('\n')

    if verbose:
        for i in data_output:
            print(i)
    return data_output
