import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

import datahandler


def main():
    ''' Train and evaluate a neural network model. '''

    # Get the training and validation data.
    x_train, y_train, x_val, y_val, test1, test2 = datahandler.load_all(10000, True)
    print('x train shape:', x_train.shape)
    print('x test shape:', x_val.shape)

    # best_test_acc = 0
    # best_dropouts = []
    # best_reg_pow = None
    #
    # for drop1 in np.linspace(0, 0.9, 10):
    #     for drop2 in np.linspace(0, 0.9, 10):
    #         for reg_pow in range(-1, -6, -1):
    #             # Train and evaluate the model.
    #             model = setup_network(x_train.shape[1:], [drop1, drop2], reg_pow)
    #             model = train_network(x_train, y_train, x_val, y_val, model, plot=False)
    #             print(f'dropouts: {[drop1, drop2]} \treg_power: {reg_pow}')
    #             print('train accuracy:', evaluate_network(model, x_train, y_train))
    #
    #             test_acc = evaluate_network(model, x_val, y_val)
    #             print('test accuracy:', test_acc)
    #
    #             # Update the best parameters so far.
    #             if test_acc > best_test_acc:
    #                 best_test_acc = test_acc
    #                 best_dropouts = [drop1, drop2]
    #                 best_reg_pow = reg_pow
    #
    #             # Show the best hyperpameters so far.
    #             print('best hyperpameters so far:')
    #             print('-------------------------')
    #             print('best test acc:', best_test_acc)
    #             print('best dropouts:', best_dropouts)
    #             print('best reg pow:', best_reg_pow)

    drop1 = 0.2
    drop2 = 0.1
    reg_pow = -6
    model = setup_network(x_train.shape[1:], [drop1, drop2], reg_pow)
    model = train_network(x_train, y_train, x_val, y_val, model, epochs=20)
    print(f'dropouts: {[drop1, drop2]} \treg_power: {reg_pow}')
    print('train accuracy:', evaluate_network(model, x_train, y_train))

    test_acc = evaluate_network(model, x_val, y_val)
    print('test accuracy:', test_acc)

    datahandler.save_model(model, "nnetwork4.model")
    datahandler.save_prediction(np.ndarray.flatten(model.predict(test1)),"data/nnetwork2008_v4")
    datahandler.save_prediction(np.ndarray.flatten(model.predict(test2)),"data/nnetwork2012_v4")
    return


def setup_network(input_shape, dropouts, reg_pow):
    ''' Create a neural network to predict probability of voting. '''

    L2 = regularizers.l2(10 ** reg_pow)

    # Setup the model.
    model = Sequential()
    model.add(Dense(400, input_shape=input_shape, activation='relu', kernel_regularizer=L2))
    model.add(Dropout(dropouts[0]))
    model.add(Dense(50, activation='relu', kernel_regularizer=L2))
    model.add(Dropout(dropouts[1]))
    model.add(Dense(50, activation='relu', kernel_regularizer=L2))
    model.add(Dropout(dropouts[1]))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model


def train_network(x, y, val_x, val_y, model=None, epochs=5, plot=True):
    ''' Setup and train a neural network. '''

    if model is None:
        # Creat the network.
        print('creating a new network ...')
        model = setup_network(x.shape[1:])

    # Train the network.
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    fit = model.fit(x, y, validation_data=(val_x, val_y), batch_size=64, epochs=epochs)
    # print('history:', fit.history.keys())

    if plot:
        x_vals = list(range(epochs))
        y_vals = np.array([fit.history['acc'], fit.history['val_acc']])
        make_plot(x_vals, y_vals)

    return model


def evaluate_network(model, x, y):
    ''' Evaluate the model on the given data using AUC loss. '''

    # Get the model predictions.
    y_pred = np.ndarray.flatten(model.predict(x))
    return datahandler.auc_err(y_pred, y)


def make_plot(x_vals, y_vals, xlabel=None, ylabel=None):
    '''
    Make a plot of y_vals accross x_vals.

    x_vals: (array) The values along the x axis.
    y_vals: (1d or 2d array) If 1d list, the values of y. If 2d list, a list
            of points for each of the lines to plot.
    xlabel: (string) The label for the x axis.
    ylabel: (string) The label for the y axis.

    Returns: None
    '''

    if len(y_vals.shape) == 1:
        # There is only one line to plot.
        plt.plot(x_vals, y_vals)
    else:
        # There are multiple lines to plot.
        for line in y_vals:
            plt.plot(x_vals, line)

    plt.show()

    return


if __name__ == '__main__':
    main()
