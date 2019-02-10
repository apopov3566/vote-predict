from datahandler import *
from pathlib import Path

n_estimators = 1000

def train_forest(data, labels, v_data, v_labels, depth, model_name):
    clf = None
    savemodel = Path(model_name)
    if savemodel.exists():
        print("model found")
        clf = load_model(model_name)
    else:
        print("training model")
        clf = RandomForestRegressor(n_estimators = n_estimators, verbose = 1, n_jobs = 10)
        clf.min_samples_leaf = depth
        clf.fit(data, labels)
        save_model(clf, model_name)


    msg = collect_model_stats(data, labels, v_data, v_labels, clf, 
        model_name, verbose = True)
    
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
    accs.append((i,train_forest(train_features, labels, 
        test_features, v_labels, i, "random_forest_pca.model")))

