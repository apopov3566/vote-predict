from datahandler import *

n_estimators = 1000

def run_forest(data, labels, v_data, v_labels, depth):

    clf = RandomForestRegressor(n_estimators = n_estimators, verbose = 1, n_jobs = -1)
    clf.min_samples_leaf = depth
    clf.fit(data, labels)

    train_result = clf.predict(data)
    validate_result = clf.predict(v_data)

    cat_result = np.floor(2 * validate_result)

    print(classification_err(cat_result, v_labels))
    print(auc_err(validate_result, v_labels))
    print(auc_err(train_result, labels))
    return auc_err(validate_result, v_labels), auc_err(train_result, labels)

data, labels, v_data, v_labels = load_train("data/train_2008.csv", 1000, 4000)
print("Load done!")
accs = []
for i in range(10,11,1):
    print(i)
    accs.append((i,run_forest(data, labels, v_data, v_labels, i)))

print(accs)
