from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import LinearSVC
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df1 = pd.read_csv("generated.csv", index_col=0, low_memory=False)
features = df1.columns
df1 = df1.interpolate(method='linear', limit_direction='forward')
df1.fillna(df1.mean(), inplace=True)

normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(df1)
X = pd.DataFrame(training, columns=features)
y = df1.rlf
# A = np.array(X.iloc[:, df1.columns != 'rlf'])
# b = np.array(X.iloc[:, df1.columns == 'rlf'])
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

print("Before OverSampling, counts of label '1': {}".format(sum(Y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train==0)))
sm = ADASYN(random_state=101)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res==0)))

classifier = LinearSVC(random_state=10)
classifier.fit(X_train_res, Y_train_res)
Y_test_prediction = classifier.predict(X_test)

# classifier = LogisticRegression()
# classifier.fit(X_train_res, Y_train_res)
# Y_test_prediction = classifier.predict(X_test)

# classifier = DecisionTreeClassifier(criterion="entropy",  max_depth=3)
# classifier.fit(X_train_res,Y_train_res)
# Y_test_prediction = classifier.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_test_prediction)
plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title('SVM - Confusion Matrix')
print(classification_report(Y_test, Y_test_prediction, target_names=["regular", "rlf"]))
score = classifier.score(X_test, Y_test)
print("Model Accuracy:{}".format(score))
