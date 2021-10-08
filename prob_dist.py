from matplotlib import pyplot
from numpy.random import normal
from numpy import mean
from numpy import std
from scipy.stats import norm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
# generate a sample
df1 = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_bin_h.csv", low_memory=False)


df = df1.std()
df.to_csv("temp.csv")
# penguins = sns.load_dataset(df1['temp_max_day1'])
# sns.displot(penguins, x="data")
# calculate parameters
# df1.to_numpy()
# sample_mean = mean(df1)
# sample_std = std(df1)
# print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
# # define the distribution
# dist = norm(sample_mean, sample_std)
# # sample probabilities for a range of outcomes
# values = [value for value in range(100000)]
# probabilities = [dist.pdf(value) for value in values]
# # plot the histogram and pdf
# pyplot.hist(df1, bins=1000, density=True)
# pyplot.plot(values, probabilities)
# pyplot.show()