# five shortest distance weather stations from radio tower

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

df1 = pd.read_csv('../input_dataset/distance/distance_matrix.csv', index_col=False, header=None)
df = pd.read_csv('../input_dataset/distance/distance.csv')
dfc = df.drop('Unnamed: 0', axis='columns')
dfr = df['Unnamed: 0']
col = list(df.columns)
ndf = pd.DataFrame()
i = 0
w = 0
distance = []
for column in dfc:
    radios = dfc.loc[dfc[column].sort_values().iloc[:] <= 23]
    # values = [x for x in radios.values if x <= 23] # taking maximum value of minimum distance
    #values = radios.values  #
    z = 0
    for s in radios.index:
        station = dfr.iloc[s]
        #distance = values[z]
        try:
            ndf.insert(0, 'site_id', [column])
            ndf.insert(1, 'station_id', [station])
            #ndf.insert(2, 'distance', [distance])
        except:
            ndf.loc[w] = [column, station]
        z += 1
        w += 1
    i += 1

ndf.to_csv("../input_dataset/distance/optimized_distance_v2.csv")

# arr = ndf['distance']
#
#
# plt.boxplot(arr)
# fig = plt.figure(figsize =(15, 12))
# plt.show()

# import seaborn as sns
#
# sns.scatterplot(x="site_id",y = 'station_id' ,data=ndf)
# plt.show()
# ax1 = ndf.plot.scatter(x = 'site_id', y = 'station_id')
# plt.show()


#
# def estimate_gaussian(dataset):
#     mu = np.mean(dataset)
#     sigma = np.std(dataset)
#     limit = sigma * 1.5
#
#     min_threshold = mu - limit
#     max_threshold = mu + limit
#
#     return mu, sigma, min_threshold, max_threshold
#
# print(max(df1))
# mu, sigma, min_threshold, max_threshold = estimate_gaussian(df1.values)
# print(min_threshold, max_threshold)
# max_threshold = 22.7327
# xs = df.columns.values
# ys = df.index
# for x, y in zip(xs, ys):
#     color = 'blue'  # non-outlier color
#     if not min_threshold <= y <= max_threshold:  # condition for being an outlier
#         color = 'red'  # outlier color
#     plt.scatter(x, y, color=color)
# plt.show()

# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(ndf['station_id'], ndf['site_id'])
# ax.set_xlabel('site id')
# ax.set_ylabel('station_id')
# plt.show()

# dx1 = ndf['distance']
# dx1 = np.array(dx1)
# n = 5
# # calculates the average
# avgResult = np.min(dx1.reshape(-1, n), axis=1)
# print(avgResult)
# ff = pd.DataFrame(avgResult)
# ff.to_csv("min_distance.csv")
# print(max(avgResult))
# # ndf['avg'] = avgResult
# # mean_result = np.mean(avgResult)
# # print(mean_result)
# #
# # ndx = ndf[ndf['distance'] < mean_result]
# #
# # ndx.to_csv('../input_dataset/Shortest_Stations_v3.csv')
