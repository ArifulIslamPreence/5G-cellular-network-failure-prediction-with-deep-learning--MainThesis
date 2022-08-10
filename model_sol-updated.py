import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from numpy import mean, std, matrix
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns
from xgboost import XGBClassifier
import lightgbm as lgb
import xgboost as xgb

dfx = pd.read_csv("../output_dataset/train_set/final_train.csv")
dft = pd.read_csv("../output_dataset/test_set/feb/final_test_feb.csv")
col = ['history_mlid_11', 'history_polarization_2', 'history_clutter_class_5']
dfx = dfx.drop(columns=col)
# print(len(dfx.columns))
# print(len(dft.columns))
# l = list()
# l1 = dfx.columns
# l3 = dft.columns
# for item in l1:
#   if item not in l3:
#     l.append(item)
#
# print(l)
#
x_train = dfx.drop('rlf', axis=1)
y_train = dfx['rlf']
y_train = np.asarray(y_train)
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30)
x_test = dft.drop('rlf', axis=1)
y_test = dft['rlf']
y_test = np.asarray(y_test)
y_test = y_test.flatten()

print(y_train.shape, y_test.shape)
ss = StandardScaler()
x_train_scale = ss.fit_transform(x_train)

#LGBM
model2 = lgb.LGBMClassifier()
model2.fit(x_train_scale, y_train)
# Actual class predictions
rf_predictions = model2.predict(x_test)
conf_matrix = confusion_matrix(y_test, rf_predictions)
rf_probs = model2.predict_proba(x_test)[:, 1]
rf_probs = pd.DataFrame(rf_probs)
rf_probs['pred'] = rf_probs

rf_probs.to_csv("../output_dataset/test_set/feb/pred_feb/pred_test_feb.csv")
# plt.figure()
# sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
# plt.title('RF Classification - Confusion Matrix')
print(classification_report(y_test, rf_predictions, target_names=["regular", "rlf"]))

#RF classifier

model1 = RandomForestClassifier(n_estimators=100,
                                bootstrap=True)

model1.fit(x_train_scale, y_train)

# Actual class predictions
rf_predictions = model1.predict(x_test)
# Probabilities for each class
rf_probs = model1.predict_proba(x_test)[:, 1]
rf_probs = pd.DataFrame(rf_probs)
rf_probs['pred'] = rf_probs

#XGB
model3 = xgb.XGBClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model3, x_train_scale, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model3.fit(x_train_scale, y_train)
rf_probs.to_csv("../output_dataset/test_set/feb/pred_feb/pred_test_feb.csv")

algorithms = [model1, model2, model3]
predictions = matrix(row_length=len(y_train), column_length=len(algorithms))

for i, algorithm in enumerate(algorithms):
    predictions[i] = algorithm.fit(x_train_scale, y_train).predict(y_test)
    rf_predictions.append(predictions[i])

from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)

conf_matrix = confusion_matrix(y_test, rf_predictions)

plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title('RF Classification - Confusion Matrix')
print(classification_report(y_test, rf_predictions, target_names=["regular", "rlf"]))

