def eval_forest(clf, data, labels, v_data, v_labels):
    train_result = clf.predict(data)
    validate_result = clf.predict(v_data)

    cat_result = np.floor(2 * validate_result)

    print(classification_err(cat_result, v_labels))
    print(auc_err(validate_result, v_labels))
    print(auc_err(train_result, labels))
    return auc_err(validate_result, v_labels), auc_err(train_result, labels)

train_data, train_labels, v_data, v_labels, test1_data, test2_data = load_all(1000)

clf = SVR(gamma='auto')
clf.fit(train_data, train_labels)

print(eval_forest(clf, train_data, train_labels, v_data, v_labels))
