# Minimum distanced weather station with rl

import csv
import numpy as np
import pandas as pd

df = pd.read_csv("../input_dataset/distance/distance.csv", index_col=0)
ws_name = []
site_name = []
min_val = []
min_val = df.idxmin()
for i, j in min_val.iteritems():
    site_name.append(i)
    ws_name.append(j)

tup = list(zip(site_name, ws_name))
df2 = pd.DataFrame(tup, columns=['site_id', 'station_id'])
val = df.min(axis=0)
dfc = pd.DataFrame(val)
dfc.to_csv("min_distance3.csv")
# df2.to_csv("../input_dataset/trainsets/siteVws.csv")
d = max(val)
print(df2.head())
print(d)
