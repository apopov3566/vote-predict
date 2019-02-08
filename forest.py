from datahandler import *

n_estimators = 1000

def train_forest(data, labels, depth, n_estimators):
    clf = RandomForestRegressor(n_estimators = n_estimators, verbose = 1, n_jobs = -1)
    clf.min_samples_leaf = depth
    clf.fit(data, labels)

    return clf

def eval_forest(clf, data, labels, v_data, v_labels):
    train_result = clf.predict(data)
    validate_result = clf.predict(v_data)

    cat_result = np.floor(2 * validate_result)

    print(classification_err(cat_result, v_labels))
    print(auc_err(validate_result, v_labels))
    print(auc_err(train_result, labels))
    return auc_err(validate_result, v_labels), auc_err(train_result, labels)

def predict_forest(clf, data, fname):
    predict_results = clf.predict(data)
    save_prediction(predict_results, fname)

data, labels, v_data, v_labels = load_train("data/train_2008.csv", 1000, 4000)

print(len(data[0]))

print("Load done!")
#m = train_forest(data, labels, 10, 1000)
#save_model(m, 'm1.model')
m = load_model('m1.model')
eval_forest(m, data, labels, v_data, v_labels)
