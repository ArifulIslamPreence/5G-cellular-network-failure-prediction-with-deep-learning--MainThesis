import argparse
import numpy as np
import pandas as pd
import time

from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

from rocket import generate_kernels, apply_kernels

data_train_col = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2_h.csv")
data_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2.csv")
target_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_v2.csv")
data_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_v2.csv")
target_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_v2.csv")

normalizer = preprocessing.MinMaxScaler()
# training = normalizer.fit_transform(x_trains)
x_trains = normalizer.fit_transform(data_train)
y_trains = target_train.to_numpy()
y_trains = y_trains.ravel()
# train_tensor = torch.tensor(training)
# train_tensor = train_tensor.float()
# target_train = target_train.to_numpy()
# train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

# Test data
x_tests = normalizer.fit_transform(data_test)
# test_tensor = torch.tensor(testing)
# test_tensor = test_tensor.float()
y_tests = target_test.to_numpy()
y_tests = y_tests.ravel()


# tscv = TimeSeriesSplit(max_train_size=None, n_splits=20)
#
#
# for train_index, test_index in tscv.split(x_trains):
#     X_train, X_test = x_trains[train_index], x_trains[test_index]
#     y_train, y_test = y_trains[train_index], y_trains[test_index]


# train test previous method

# def train_eval_split(df: pd.DataFrame, column: str, train_ratio: float = 0.7):
#     if train_ratio > 1:
#         raise ValueError(f'train ration cannot be {train_ratio}')
#
#     train_df = pd.DataFrame([], columns=df.columns)
#     test_df = pd.DataFrame([], columns=df.columns)
#     for c in set(df[column]):
#         temp = df.loc[df[column] == c]
#         # temp = temp.sort_values(by='datetime').reset_index(drop=True)
#         temp = temp.sample(temp.shape[0]).reset_index(drop=True)
#
#         l = temp.shape[0]
#         n = int(l * train_ratio)
#
#         train_df = pd.concat([train_df, temp.iloc[:n]], axis=0)
#         test_df = pd.concat([test_df, temp.iloc[n:]], axis=0)
#
#     return train_df, test_df


def run(x_train, y_train, x_test, y_test, num_runs=100, num_kernels=500):
    results = np.zeros(num_runs)
    pred_score = []

    Y_training, X_training = y_train, x_train
    Y_test, X_test = y_test, x_test

    for i in range(num_runs):
        input_length = X_training.shape[1]
        kernels = generate_kernels(input_length, num_kernels)

        X_training_transform = apply_kernels(X_training, kernels)

        X_test_transform = apply_kernels(X_test, kernels)

        classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
        classifier.fit(X_training_transform, Y_training)

        results[i] = classifier.score(X_test_transform, Y_test)
        pred_score.append(classifier.predict(X_test_transform))
    return results, pred_score


_ = generate_kernels(100, 500)
apply_kernels(x_trains, _)

results, pred_score = run(x_trains, y_trains, x_tests, y_tests,
                          num_runs=100,
                          num_kernels=500)

res = pd.DataFrame(results)
res.to_csv("../output_dataset/train_set/rocket_results_v2.csv")
pred = pd.DataFrame(pred_score)
pred.to_csv("../output_dataset/train_set/rocket_pred_scores.csv")
