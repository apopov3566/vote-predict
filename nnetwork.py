import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

import datahandler


def main():
    ''' Train and evaluate a neural network model. '''

    # Get the training and test data.
    x_train, y_train, x_test, y_test = datahandler.load_train("data/train_2008.csv", 1000, 4000)
    print('x train shape:', x_train.shape)
    print('x test shape:', x_test.shape)

    # Train and evaluate the model.
    model = train_network(x_train, y_train, x_test, y_test)
    print('train accuracy:', evaluate_network(model, x_train, y_train))
    print('test accuracy:', evaluate_network(model, x_test, y_test))

    return


def setup_network(input_shape):
    ''' Create a neural network to predict probability of voting. '''

    L2 = regularizers.l2(0.01)

    # Setup the model.
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation='relu', kernel_regularizer=L2))
    model.add(Dropout(0.7))
    model.add(Dense(75, activation='relu', kernel_regularizer=L2))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_network(x, y, val_x, val_y, epochs=50, plot=True):
    ''' Setup and train a neural network. '''

    # Creat the network.
    model = setup_network(x.shape[1:])
    model.summary()

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
    ''' Evaluate the model on the given data using 0/1 loss. '''

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
