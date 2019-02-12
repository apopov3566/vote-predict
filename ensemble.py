import numpy as np

import datahandler


def evaluate(model, x, y):
    ''' Evaluate the model on the given data using AUC loss. '''

    # Get the model predictions.
    y_pred = np.ndarray.flatten(model.predict(x))
    return datahandler.auc_err(y_pred, y)


def get_predictions(models, weights, x):
    ''' Get the ensemble predictions by weighting each model prediction. '''

    predictions = np.zeros(x.shape[0])

    for i in range(len(models)):
        predictions += weights[i] * np.ndarray.flatten(models[i].predict(x))

    return predictions / sum(weights)


# Load the models.
model_files = ['nnetwork1.model', 'nnetwork2.model', 'nnetwork3.model']
models = [datahandler.load_model(filename) for filename in model_files]
print('models loaded')

# Load the data and split the validation set into two validation sets.
x_train, y_train, x_val, y_val, test1, test2 = datahandler.load_all(10000, True)
val_size = y_val.shape[0] // 2
x_val2 = x_val[val_size:]
y_val2 = y_val[val_size:]
x_val = x_val[:val_size]
y_val = y_val[:val_size]
print('x train shape:', x_train.shape)
print('x test shape:', x_val.shape)

# Evaluate each of the models to get their weights for prediction.
model_val_accs = [evaluate(model, x_val, y_val) for model in models]
for acc in model_val_accs:
    print('model accuracy:', acc)

predictions = get_predictions(models, model_val_accs, x_val2)
print('ensemble validation accuracy:', datahandler.auc_err(predictions, y_val2))

# Save the predictions
datahandler.save_prediction(get_predictions(models, model_val_accs, test1),"data/nnetwork2008_ensemble")
datahandler.save_prediction(get_predictions(models, model_val_accs, test2),"data/nnetwork2012_ensemble")
