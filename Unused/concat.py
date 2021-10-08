#combining dataset using merge command

import csv
import numpy as np
import pandas as pd

df1 = pd.read_csv("../Dataset/rl-kpis.csv", index_col=0,low_memory=False)
df2 = pd.read_csv("../Dataset/met-real-temp.csv", index_col=0,low_memory=False)
df3 = pd.read_csv("../Dataset/siteVws.csv",index_col=0)


df4 = pd.merge(df2,df3,how="inner",left_on='station_no',right_on='station_id')


dfx = pd.merge(df1,df4, how="inner",left_on=['site_id','datetime'],right_on=['site_id','datetime'])
dfx["scalibility_score"].replace(np.NaN,dfx["scalibility_score"].mean().tail())
#dfx.to_csv("combined_short-main.csv")
print(dfx.head())