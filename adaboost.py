import numpy as np
from sklearn.ensemble import AdaBoostClassifier

import datahandler


# Load the trained model.
# clf = datahandler.load_model('adaboost.model')

# Get the training and validation data.
x_train, y_train, x_val, y_val, _, _ = datahandler.load_all(10000, True)
print('x train shape:', x_train.shape)
print('x test shape:', x_val.shape)

# Train the model.
clf = AdaBoostClassifier()
clf.fit(x_train, y_train)

datahandler.save_model(clf, 'adaboost.model')


def evaluate(model, x, y):
    ''' Evaluate the model on the given data using AUC loss. '''

    # Get the model predictions.
    y_pred = np.ndarray.flatten(model.predict(x))
    return datahandler.auc_err(y_pred, y)


# Get the training and validation accuracy.
print('train accuracy:', evaluate(clf, x_train, y_train))
print('test accuracy:', evaluate(clf, x_val, y_val))
