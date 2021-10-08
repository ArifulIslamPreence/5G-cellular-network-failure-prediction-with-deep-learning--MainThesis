import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy
from datetime import date, timedelta

# History kpi
#
# df1 = pd.read_csv("../input_dataset/testsets/march/rl-kpis_3.csv", index_col=0, low_memory=False)
# df2 = pd.read_csv("../input_dataset/testsets/rl-sites.csv")
# df3 = pd.read_csv('../output_dataset/train_set/min_max_distanced/combined_forecast.csv')


# kpis history data

# rl_kpis_history = df1.merge(df2[['site_id', 'groundheight', 'clutter_class']], on='site_id')
# rl_kpis_history.columns = ['history_{}'.format(column) for column in rl_kpis_history.columns]
# rl_kpis_history['history_datetime'] = [pd.Timestamp(x) for x in rl_kpis_history['history_datetime']]
# rl_kpis_history.sort_values(by='history_datetime')
# # rl_kpis_history.to_csv("../input_dataset/testsets/march/rl-kpis_history_march.csv")
#
# # rl kpis Modification
# rl_kpis_mod = df1.copy()
# rl_kpis_mod['datetime'] = [pd.Timestamp(x) for x in rl_kpis_mod['datetime']]
# rl_kpis_mod = rl_kpis_mod.drop(rl_kpis_mod.index[0:2691])
# rl_kpis_mod['forecast_datetime'] = [x - pd.Timedelta(days=1) for x in rl_kpis_mod['datetime']]
# rl_kpis_mod.to_csv("../input_dataset/testsets/march/rl-kpis_3mod.csv")

# kpis's merging
df3 = pd.read_csv('../output_dataset/train_set/outlier_removed/combined_optimized_ol.csv')
df4 = pd.read_csv('../input_dataset/trainsets/rl-kpis_history.csv', low_memory=False)
df3['forecast_datetime'] = [pd.Timestamp(x) for x in df3['forecast_datetime']]
df4['history_datetime'] = [pd.Timestamp(x) for x in df4['history_datetime']]
df4.sort_values(by='history_datetime')

final_merged = df3.merge(df4, left_on=['mlid', 'forecast_datetime'],
                         right_on=['history_mlid', 'history_datetime'])
# # # final_merged = pd.merge(left=df3, right=df4, how='outer', left_on=['mlid','forecast_datetime'], right_on=[
# # # 'history_mlid','history_datetime'])
final_merged.to_csv("../output_dataset/train_set/outlier_removed/history_merged_ol.csv")
#
# dropping features from next day keeping only rlf

df3 = final_merged
col = ['Column1', 'type', 'datetime', 'tip', 'mlid', 'mw_connection_no', 'site_id', 'neid', 'direction', 'polarization',
       'card_type', \
       'adaptive_modulation', 'freq_band', 'link_length', 'severaly_error_second', 'error_second', 'unavail_second',
       'avail_time', 'bbe', \
       'rxlevmax', 'scalibility_score', 'capacity', 'modulation', 'forecast_datetime']

df3 = df3.drop(columns=col)
print(df3.columns)
df3.to_csv("../output_dataset/train_set/outlier_removed/final_merge_ol.csv")
