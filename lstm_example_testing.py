
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import pandas as pd
from matplotlib import pyplot
import numpy as np

traindf = pd.read_csv('../output_dataset/train_set/min_max_distanced/encoded_train_optimised_one.csv')
# testdf = pd.read_csv('../output_dataset/test_set/feb/final_test_one-feb.csv')

x_train = traindf.drop('rlf', axis=1)
y_train = traindf['rlf']
number_of_features = len(x_train.columns)

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20)

x_trains = X_train[:].values
y_trains = y_train[:].values
y_trains = y_trains.reshape(1, -1)

# x_test = testdf.drop('rlf', axis=1)
# y_test = testdf['rlf']

x_tests = X_test[:].values
y_tests = y_test[:].values
y_tests = y_tests.reshape(1, -1)


# tscv = TimeSeriesSplit()

# TimeSeriesSplit(max_train_size=None, n_splits=11)
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]


# scale train and test data to [-1, 1]
def scale(x_train, x_test):
    # fit scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(x_train)
    # transform train

    x_train_scaled = scaler.transform(x_train)

    # transform test
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


# fit an LSTM network to training data
def fit_lstm(x_train, y_train, batch_size, nb_epoch, neurons):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x_train, y_train), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, x_test):
    X = x_test
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


x_train_scaled,x_test_scaled =scale(x_trains, x_tests)

# fit the model
lstm_model = fit_lstm(x_train_scaled,y_trains, 3000,100, 4)
# forecast the entire training dataset to build up state for forecasting

lstm_model.predict(x_test_scaled, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(x_test_scaled)):
    # make one-step forecast
    X, y = x_test_scaled,y_tests
    yhat = forecast_lstm(lstm_model, 1, X)

    # store forecast
    predictions.append(yhat)



# report performance
rmse = sqrt(mean_squared_error(y_tests, predictions))
print('Test RMSE: %.3f' % rmse)
dfa = pd.DataFrame(y_tests,predictions, columns=["actual", "prediction_score","predicted values"])
dfa.to_csv("results.csv")
# line plot of observed vs predicted
pyplot.plot(y_tests)
pyplot.plot(predictions)
pyplot.show()
