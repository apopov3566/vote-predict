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

train_data, train_labels, v_data, v_labels, test1_data, test2_data = load_all(1000)
print(len(train_data[0]))

print("Load done!")
#m = train_forest(train_data, train_labels, 15, 2000)
#save_model(m, 'submit1.model')
m = load_model('submit1.model')
eval_forest(m, train_data, train_labels, v_data, v_labels)
predict_forest(m, test1_data, "data/2008sub1.csv")
predict_forest(m, test2_data, "data/2012sub1.csv")
