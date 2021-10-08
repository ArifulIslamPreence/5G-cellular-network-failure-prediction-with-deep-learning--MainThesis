import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import math
import matplotlib.pyplot as plt
import time_cv
import datetime
from datetime import datetime as dt
from sklearn.model_selection import TimeSeriesSplit

train_data = pd.read_csv("../output_dataset/train_set/outlier_removed/history_merged_ol.csv",low_memory=False,error_bad_lines = False, parse_dates=['datetime'])
df = pd.read_csv("../output_dataset/train_set/outlier_removed/final_data_ol_en.csv",low_memory=False,error_bad_lines = False)
df['datetime'] = train_data['datetime']



# Time series split of scikit-learn library

# tscv = TimeSeriesSplit()
# TimeSeriesSplit(n_splits=18)
# for train_index, test_index in tscv.split(X):
#      print("TRAIN:", train_index, "TEST:", test_index)
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
#
# print(X_train)

# Time series split of customized algorithm
# ref : https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8

tscv = time_cv.TimeBasedCV(train_period=120,
                           test_period=30,
                           freq='days')
for train_index, test_index in tscv.split(df,
                                          validation_split_date=None, date_column='datetime'):
    continue

# # get number of splits
# tscv.get_n_splits()

# Perform splitting by index

for train_index, test_index in tscv.split(df, validation_split_date=None):
    data_train = df.loc[train_index].drop('datetime', axis=1)
    data_test = df.loc[test_index].drop('datetime', axis=1)





#
train_x = pd.DataFrame(data_train)

test_x = pd.DataFrame(data_test)

# dropping failed link

train_x = train_x[train_x['rlf'] < 1]

# train_y.values.flatten()
# test_y.values.flatten()
# print(test_y.head())
#
train_x.to_csv("../output_dataset/cv_train_test/outlierfree/updated/train_data.csv", header=False)


test_x.to_csv("../output_dataset/cv_train_test/outlierfree/updated/validation_data.csv", header=False)


# CV results
#
# train_x = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_bin.csv")
# train_y = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_bin.csv")
# test_x = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_bin.csv")
# test_y = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_bin.csv")
#
# train_y = train_y.values.ravel()
# test_y = test_y.values.ravel()
#
#
# def merge(list1, list2):
#     merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
#     return merged_list
#
#
# from sklearn.linear_model import LogisticRegression
#
# clf = LogisticRegression(solver='lbfgs', max_iter=1000)
# clf.fit(train_x, train_y)
# scores = []
# preds = clf.predict(test_x)
# conf_matrix = confusion_matrix(test_y, preds)
# # accuracy for the current fold only
# r2score = clf.score(test_x, test_y)
# scores.append(preds)
# dft = pd.DataFrame(scores)
# dft = dft.T
# dft['true'] = test_y
# dft.to_csv("../output_dataset/cv_train_test/outlierfree/cv_pred_ol3.csv")
# # this is the average accuracy over all folds
# average_r2score = np.mean(scores)
# print(scores)
# print(conf_matrix)
# # print(classification_report(test_x, preds, target_names=[0, 1]))
# # print(average_r2score)
