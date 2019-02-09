from datahandler import *

n_estimators = 1000

def run_forest(data, labels, v_data, v_labels, depth):

    clf = RandomForestRegressor(n_estimators = n_estimators, verbose = 1, n_jobs = 10)
    clf.min_samples_leaf = depth
    clf.fit(data, labels)

    msg = collect_model_stats(data, labels, v_data, v_labels, clf, 
        "random forest_pca" + str(n_estimators), verbose = True)
    
    return msg

data, labels, v_data, v_labels = load_train("data/train_2008.csv", 1000, 4000)
print("Load done!")

# modify dataset with PCA

pca = PCA(n_components = 500)
pca.fit(data)
train_features = pca.transform(data)
test_features = pca.transform(v_data)

accs = []
for i in range(10,11,1):
    print(i)
    accs.append((i,run_forest(train_features, labels, test_features, v_labels, i)))

