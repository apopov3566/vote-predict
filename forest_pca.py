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

def pca_modifier(n_eigen_vectors, depth, data, labels, v_data, v_labels):
    
    pca = PCA(n_components = n_eigen_vectors)
    pca.fit(data)
    train_features = pca.transform(data)
    test_features = pca.transform(v_data)
    train_forest(train_features, labels,
            test_features, v_labels, depth, str(n_eigen_vectors) + "random_forest_pca.model")


data, labels, v_data, v_labels = load_train("data/train_2008.csv", 10000, 40000)
print("Load done!")

for j in [i * 125 + 500 for i in range(10)]:
    for depth in range(10,11,1):
        pca_modifier(j, depth, data, labels, v_data, v_labels)


# accs = []
# for i in range(10,11,1):
#     print(i)
#     for j in [i * 125 + 500 for i in range(10)]:
#         accs.append((i,train_forest(train_features, labels,
#             test_features, v_labels, i, "random_forest_pca.model")))

# test commit