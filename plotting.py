import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# dfx = pd.read_csv('../input_dataset/distance/distance.csv')
# df = pd.read_csv('../input_dataset/distance/distance_matrix.csv')

df = pd.read_csv('../input_dataset/distance/distance.csv')
X = df.to_numpy()
col = list(df.columns)
ax = df.hist(column= col, bins=25, grid=False, figsize=(25,22), color='#86bf91' )



# Set x-axis label
ax.set_xlabel("Session Duration (Seconds)", labelpad=20, weight='bold', size=12)


ax.set_ylabel("Sessions", labelpad=20, weight='bold', size=12)


# site = list(dfx.columns)
# station = list(dfx.iloc[:, ])
# epsilon = 23
#
# db = DBSCAN(eps=epsilon, min_samples = 117).fit(X)
# labels = db.labels_
#
# no_clusters = len(np.unique(labels))
# no_noise = np.sum(np.array(labels) == -1, axis=0)
#
# print('Estimated no. of clusters: %d' % no_clusters)
# print('Estimated no. of noise points: %d' % no_noise)
#
# # Generate scatter plot for training data
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
# plt.title('Two clusters with data')
# plt.xlabel('site id')
# plt.ylabel('weather id')
# plt.show()
