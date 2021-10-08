import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy

df = pd.read_csv("../output_dataset/train_set/pred/pred_train.csv")


ls2 = list()
lss = list()
ls = df['pred']

for x in ls:
    x = x + 0.07
    ls2.append(x)

df['pred'] = ls2

for i in ls2:
    if i < 0.5:
        lss.append('FALSE')
    else:
        lss.append('TRUE')
df['rlf'] = lss
df.to_csv('../output_dataset/train_set/pred/pred_xgb.csv')