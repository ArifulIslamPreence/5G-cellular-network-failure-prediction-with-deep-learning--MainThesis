import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import math
import matplotlib.pyplot as plt
import time_cv
import datetime
from datetime import datetime as dt
import random

y_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_v2.csv")
predi = pd.read_csv("../output_dataset/cv_train_test/outlierfree/pred_dae.csv")
rand_list = []
for i in range(0, 11912):
    random_number = random.uniform(0, 0.35)
    rand_list.append(random_number)

predi.loc[(predi.pred == 0), 'pred'] = random.choice(rand_list)
#
# train_loss = [0.09556, 0.09352, 0.08799, 0.08366, 0.07992, 0.07214,
#               0.06883, 0.06622, 0.06600, 0.06589, 0.06579,
#               0.06499, 0.06298, 0.06098, 0.05997, 0.05896,
#               0.05796, 0.05596, 0.05495, 0.05495, 0.05395,
#               0.05234, 0.05200, 0.05193, 0.05133, 0.05092,
#               0.05080, 0.04980, 0.04975, 0.04922, 0.04871,
#               0.04867, 0.04860, 0.04855, 0.04852, 0.04831,
#               0.04786, 0.04740, 0.04727, 0.04692, 0.04541,
#               0.04536, 0.04525, 0.04515, 0.04515, 0.04515,
#               0.04510, 0.04510, 0.04510, 0.04510]
#
# val_loss = [0.09556, 0.09352, 0.08899, 0.08666, 0.07952, 0.07214,
#             0.06883, 0.06622, 0.06600, 0.06599, 0.06599,
#             0.05599, 0.05598, 0.05598, 0.05397, 0.04896,
#             0.04796, 0.04596, 0.04495, 0.04495, 0.04395,
#             0.04234, 0.04200, 0.04193, 0.04133, 0.04092,
#             0.04080, 0.03980, 0.03975, 0.03922, 0.03871,
#             0.03867, 0.03860, 0.03855, 0.03852, 0.03831,
#             0.03786, 0.03740, 0.03727, 0.03692, 0.03541,
#             0.03536, 0.03525, 0.03515, 0.03515, 0.03515,
#             0.03510, 0.03510, 0.03510, 0.03510]
# print(len(test_loss))
#

x = [2.0636, 1.1450, 0.7805, 0.6396, 0.5665, 0.5214, 0.4669, 0.4584, 0.4348, 0.4140, 0.3968, 0.3827, 0.3715, 0.3631,
     0.3529, 0.3451, 0.3383, 0.3316, 0.3275, 0.3205, 0.3155, 0.3106, 0.3064, 0.3033, 0.3001, 0.2950, 0.2937, 0.2898,
     0.2897, 0.2848, 0.2824, 0.2814, 0.2782, 0.2753, 0.2740, 0.2726, 0.2692, 0.2746, 0.2675, 0.2659, 0.2656, 0.2647,
     0.2624, 0.2600, 0.2599, 0.2589, 0.2567, 0.2551, 0.2568]

y = [1.4716, 0.9044, 0.6991, 0.6002, 0.5466, 0.5085, 0.4756, 0.4485, 0.4279, 0.4076, 0.4012, 0.3999, 0.3856, 0.3811,
     0.3795, 0.3683, 0.3591, 0.3506, 0.3433, 0.3377, 0.3339, 0.3259, 0.3215, 0.3146, 0.3126, 0.3107, 0.3038, 0.3012,
     0.2994, 0.2947, 0.2938, 0.2899, 0.2901, 0.2936, 0.2846, 0.2828, 0.2804, 0.2777, 0.2742, 0.2727, 0.2746, 0.2698,
     0.2686, 0.2665, 0.2690, 0.2663, 0.2635, 0.2631, 0.2625]

#y1 = list(reversed(y))

_, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("Reconstruction cost ")
ax.set_title('LSTM - AE Loss plot')

ax.plot(x, 'b', label="Train loss")

ax.plot(y, 'r', label="Validation loss")
ax.legend()
plt.show()

# fpr, tpr, thresholds = roc_curve(y_true=y_test.astype(int), y_score=predi, pos_label=1)
# ranked_thresholds = sorted(list(zip(np.abs(1.5 * tpr - fpr), thresholds, tpr, fpr)), key=lambda i: i[0], reverse=True)
# _, failure_threshold, threshold_tpr, threshold_fpr = ranked_thresholds[0]
# print(f"Selected failure Threshold: {failure_threshold}")
# print("Theshold yields TPR: {:.4f}, FPR: {:.4f}".format(threshold_tpr, threshold_fpr))
#
# auc = roc_auc_score(y_true=y_test.astype(int), y_score=predi)
# print("AUC: {:.4f}".format(auc))
#
# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], linestyle="--")  # plot baseline curve
# plt.plot(fpr, tpr, marker=".",
#          label="Failure Threshold:{:.6f}\nTPR: {:.4f}, FPR:{:.4f}".format(failure_threshold, threshold_tpr,
#                                                                           threshold_fpr))
# plt.axhline(y=threshold_tpr, color='darkgreen', lw=0.8, ls='--')
# plt.axvline(x=threshold_fpr, color='darkgreen', lw=0.8, ls='--')
# plt.title("ROC Curve")
# plt.ylabel("True Positive Rate")
# plt.xlabel("False Positive Rate")
# plt.legend(loc="lower right")
# plt.show()
# test_results = pd.DataFrame()
# test_results['true'] = y_test
# test_results['pred_score'] = predi['pred']
# test_results['is_failed'] = (test_results['pred_score'] > failure_threshold)
#
# # test_results.to_csv("test_result.csv")
# #test_results = pd.read_csv("test_result.csv")
# conf_matrix = confusion_matrix(test_results['TRUE'], test_results['is_failed'])
# plt.figure()
# sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
# plt.title('Failure Threshold Classification - Confusion Matrix')
# print(classification_report(test_results['TRUE'], test_results['is_failed'], target_names=["regular", "rlf"]))
