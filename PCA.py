import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df1 = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_bin.csv", low_memory=False)

features = len(df1.columns)
X = StandardScaler().fit_transform(df1)
# PCA applied

pca_data = PCA(n_components= 0.95)
pca_data.fit(X)
pca_reduced = pca_data.transform(X)
x,n = pca_reduced.shape
pca_Df = pd.DataFrame(data=pca_reduced
                       , columns=['Component {}'.format(i) for i in range(n)])


# pca_Df.to_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_pca_bin.csv")
print(pca_Df)