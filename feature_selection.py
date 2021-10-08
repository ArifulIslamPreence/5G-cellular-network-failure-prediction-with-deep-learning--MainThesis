from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy

dfx = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2_h.csv", low_memory=False)
dfy = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_v2.csv", low_memory=False)



ss = StandardScaler()
x_train_scale = ss.fit_transform(dfx)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_scale, dfy)
importances = pd.DataFrame(data={
    'Attribute': dfx.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)

df1 = pd.DataFrame(importances,columns=['features','importance'])
less_imp = df1["features"][-0.05 <= df1["importance"] <= 0.05]
print(df1.head())
# df1.to_csv("../output_dataset/cv_train_test/outlierfree/feature_imp_onehot.csv")
# iteratively removing features with coeff scores close to zero and doesnt affect F1 score

# less_Important_features = ['history_mlid_5', 'history_tip_1', 'history_mlid_10', 'history_error_second',
#                            'history_clutter_class_0', 'history_freq_band_0', 'history_card_type_0', 'history_mlid_0',
#                            'history_polarization_0', 'history_modulation_0', 'weather_day1_0', 'history_mlid_6',
#                            'history_modulation_3', 'history_severaly_error_second', 'history_clutter_class_1',
#                            'history_tip_0', 'history_mw_connection_no','Unnamed: 0']
#
#
# dfx = dfx.drop(columns=less_Important_features)
# dfx = ss.fit_transform(dfx)
# dfx = pd.DataFrame(dfx)
#dfx.to_csv("../output_dataset/train_set/min_max_distanced/final_train.csv")