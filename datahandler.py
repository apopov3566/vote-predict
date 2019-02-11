import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import compose
from sklearn.decomposition import PCA
import datetime
from category_handler import get_attribute_list
import category_encoders as ce

def load_all(n_validate, verbose=False):
    '''loads, splits, and changes categorical data to categories for
    training data'''

    if verbose:
        print("load...")
    train_all = np.genfromtxt("data/train_2008.csv", dtype=float, delimiter=',', names=True)
    test_1 = np.genfromtxt("data/test_2008.csv", dtype=float, delimiter=',', names=True)
    test_2 = np.genfromtxt("data/test_2012.csv", dtype=float, delimiter=',', names=True)

    if verbose:
        print("shuffle...")
    np.random.shuffle(train_all)

    if verbose:
        print("get categories...")
    cont_names = get_attribute_list("cat_continuous.dat")
    cat_names = get_attribute_list("cat_categorical.dat")
    uns_names = get_attribute_list("cat_unsure.dat")
    irr_names = get_attribute_list("cat_irrelevant.dat")

    # get + format categorical data
    train_cat = np.array(train_all[cat_names].tolist())
    test1_cat = np.array(test_1[cat_names].tolist())
    test2_cat = np.array(test_2[cat_names].tolist())

    train_len = len(train_cat)
    test1_len = len(test1_cat)
    test2_len = len(test2_cat)

    if verbose:
        print(train_len, test1_len, test2_len)
        print(len(train_cat[0]), len(test1_cat[0]), len(test2_cat[0]))

    full_cat = np.concatenate((train_cat, test1_cat))
    full_cat = np.concatenate((full_cat, test2_cat))

    if verbose:
        print(len(full_cat), len(full_cat[0]))

    full_cat = make_categorical(full_cat)
    print(full_cat.shape)

    train_cat = full_cat[:train_len]
    test1_cat = full_cat[train_len:train_len + test1_len]
    test2_cat = full_cat[train_len + test1_len:train_len + test1_len + test2_len]

    if verbose:
        print("get continuous...")
    # get + format continuous data
    train_cont = make_regularized(np.array(train_all[cont_names].tolist()))
    test1_cont = make_regularized(np.array(test_1[cont_names].tolist()))
    test2_cont = make_regularized(np.array(test_2[cont_names].tolist()))

    if verbose:
        print("zip...")
    # zip data

    print(train_cat.shape, train_cont.shape)
    print(train_cat)
    print(train_cont)
    train_data = np.hstack((train_cat, train_cont))
    test1_data = np.hstack((test1_cat, test1_cont))
    test2_data = np.hstack((test2_cat, test2_cont))

    train_labels = train_all["target"]

    v_data = train_data[:n_validate]
    v_labels = train_labels[:n_validate]
    train_data = train_data[n_validate:]
    train_labels = train_labels[n_validate:]

    # return data
    return train_data, train_labels, v_data, v_labels, test1_data, test2_data

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


def reg(data):
    """regularizes column"""
    data = data.astype(np.float)
    if np.std(data) > 0:
        return (data - np.average(data))/(np.std(data))
    return data - np.average(data)

def make_regularized(data):
    """regularizes all data"""
    X = np.insert(np.apply_along_axis(reg,0,data), 0, 1, axis=1)
    return X

def make_categorical(data):
    enc =  preprocessing.OneHotEncoder(categories='auto')
    return enc.fit_transform(data).todense()

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
