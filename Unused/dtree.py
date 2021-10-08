import pydotplus as pydotplus
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix as cfmat, classification_report, accuracy_score, precision_score, \
    recall_score, f1_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import torch
from IPython.display import Image
import torch.optim as optim

# for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import six
from six import StringIO
import sys

sys.modules['sklearn.externals.six'] = six

df1 = pd.read_csv("output_dataset/new_combined.csv", index_col=0, low_memory=False)
features = df1.columns
df1 = df1.interpolate(method='linear', limit_direction='forward')
df1.fillna(df1.mean(), inplace=True)

normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(df1)
X = pd.DataFrame(training, columns=features)
y = df1.rlf

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

# dtree = DecisionTreeClassifier()
# dtree.fit(X_train, Y_train)
# y_predict = dtree.predict(X_test)
#
# cfmat(Y_test, y_predict)
#
# print(classification_report(Y_test, y_predict))

clf = DecisionTreeClassifier(criterion="entropy",  max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('changed_decision_tree.png')
Image(graph.create_png())
