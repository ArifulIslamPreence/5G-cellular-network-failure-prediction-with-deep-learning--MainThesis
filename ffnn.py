
# fULLY CONNECTED NEURAL NET MODEL. prediction results used as a baseline 

# NOT FINAL VERSION 

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
import torch
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import pandas as pd
import copy
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_train_col = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2_h.csv")
data_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2.csv")
target_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_v2.csv")
data_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_v2.csv")
target_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_v2.csv")

normalizer = preprocessing.StandardScaler()
# training = normalizer.fit_transform(x_trains)
training = normalizer.fit_transform(data_train)
train_tensor = torch.tensor(training)
train_tensor = train_tensor.float()
target_train = target_train.to_numpy()
#train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

# Test data
testing = normalizer.fit_transform(data_test)
test_tensor = torch.tensor(testing)
test_tensor = test_tensor.float()
test_X = testing
test_Y = target_test.to_numpy()


#

def create_ffnn():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_ffnn, epochs=100, batch_size=3000, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, training, target_train, cv=kfold)

print(results)
print("\n \n")
print("results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
