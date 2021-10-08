import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy
from datetime import date, timedelta
import category_encoders as ce

df = pd.read_csv("../output_dataset/train_set/outlier_removed/final_merge_filled_ol.csv")
print(df.columns)
# Manually have to change the 'rlf' and 'history_rlf' column value from FALSE to 0 and  TRUE to 1. Cant be done by
# program due to limited memory issue

# categorical features

column = ['weather_day1', 'history_mlid', 'history_type', 'history_tip', 'history_direction', 'history_polarization',
          'history_card_type', 'history_adaptive_modulation', 'history_freq_band', 'history_modulation',
          'history_clutter_class']

# Binary encoding

# encoder = ce.BinaryEncoder(
#     cols=column, return_df=True)
# data_encoded = encoder.fit_transform(df)

# one hot encoding
data_encoded = pd.get_dummies(df, columns=column)

encoded_train_data = pd.DataFrame(data_encoded)
# encoded_train_data["rlf"] = encoded_train_data["rlf"].astype(str)
# encoded_train_data["history_rlf"] = encoded_train_data["history_rlf"].astype(str)
# encoded_train_data["rlf"] = encoded_train_data["rlf"].replace({"FALSE": 0, "TRUE": 1}, inplace=True)
# encoded_train_data["history_rlf"] = encoded_train_data["history_rlf"].replace({"FALSE": 0, "TRUE": 1}, inplace=True)
encoded_train_data['history_scalibility_score'] = pd.to_numeric(encoded_train_data["history_scalibility_score"],
                                                                 downcast="float")

encoded_train_data.to_csv("../output_dataset/train_set/outlier_removed/final_data_ol_en.csv")

# encoded_train_data.to_csv("../output_dataset/train_set/min_max_distanced/encoded_train_optimised_binary.csv")
