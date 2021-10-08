import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy
from datetime import date, timedelta

# filling missing data


df2 = pd.read_csv("../output_dataset/train_set/outlier_removed/final_merge_ol.csv")

df2['history_neid'].fillna('None', inplace=True)
df2['history_direction'].fillna('None', inplace=True)
df2['history_polarization'].fillna('None', inplace=True)
df2['history_link_length'].fillna(0, inplace=True)
df2['history_scalibility_score'].fillna(0, inplace=True)
df2['history_freq_band'].fillna(method='ffill', inplace=True)
df2['humidity_max_day1'].fillna(int(df2['humidity_max_day1'].mean()), inplace=True)
df2['humidity_min_day1'].fillna(int(df2['humidity_min_day1'].mean()), inplace=True)
df2['wind_dir_day1'].fillna(int(df2['wind_dir_day1'].mean()), inplace=True)
df2['wind_speed_day1'].fillna(int(df2['wind_speed_day1'].mean()), inplace=True)

col = ['station_no', 'datetime_ws', 'history_datetime',
       'history_site_id', 'history_neid', 'Unnamed: 0']
df2 = df2.drop(columns=col)

df2.to_csv("../output_dataset/train_set/outlier_removed/final_merge_filled_ol.csv")
