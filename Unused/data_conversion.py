
import csv
import numpy as np
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("../input_dataset/combined_short-main.csv", index_col=0, low_memory=False)

df = df1.fillna(0)


def string_conversion(x):
    if x == 0:
        return x
    elif x.isnumeric():
        return int(x)
    elif x == 'ENK':
        return 1
    elif x == 'NEC':
        return 2
    elif x == 'NEAR':
        return 1
    elif x == 'FAR':
        return 2
    elif x == 'NEAR END':
        return 1
    elif x == 'cardtype1':
        return 1
    elif x == 'cardtype2':
        return 2
    elif x == 'cardtype3':
        return 3
    elif x == 'cardtype4':
        return 4
    elif x == 'f1':
        return 1
    elif x == 'f2':
        return 2
    elif x == 'f3':
        return 3
    elif x == 'Horizontal':
        return 1
    elif x == 'Vertical':
        return 2
    elif x == '512QAM(QO)':
        return 512
    elif x == '128QAM':
        return 128
    elif x == '256QAM(QO)':
        return 256
    elif x == 'C-  PSK':
        return 1
    elif x == 'FALSE':
        return 0
    elif x == 'TRUE':
        return 1
    return 0


def datetime_conversion(d):
    return d.timestamp()


df['type'] = df['type'].apply(string_conversion)
df['tip'] = df['tip'].apply(string_conversion)
df['direction'] = df['direction'].apply(string_conversion)
df['card_type'] = df['card_type'].apply(string_conversion)
df['rlf'] = df['rlf'].apply(string_conversion)
df['freq_band'] = df['freq_band'].apply(string_conversion)
df['polarization'] = df['polarization'].apply(string_conversion)
df['modulation'] = df['modulation'].apply(string_conversion)

df['datetime'] = df['datetime'].apply(datetime_conversion)

unused_features = ['mild', 'station_id_x']

df.drop(unused_features, axis=1)
df.to_csv("combined_short_main-num.csv")
