from datahandler import *

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



# load data
train_data, train_labels, v_data, v_labels, test1_data, test2_data = load_all(0)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_data, train_labels)

#m = train_forest(train_data, train_labels, 15, 2000)
save_model(rf_random, 'submit2.model')
#m = load_model('submit1.model')
eval_forest(rf_random, train_data, train_labels, v_data, v_labels)
predict_forest(rf_random, test1_data, "data/2008sub1.csv")
predict_forest(rf_random, test2_data, "data/2012sub1.csv")
