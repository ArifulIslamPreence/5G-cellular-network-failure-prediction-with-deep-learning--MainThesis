from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from rocket import generate_kernels, apply_kernels
df1 = pd.read_csv("output_dataset/new_combined.csv", index_col=0, low_memory=False)
features = df1.columns
df1 = df1.interpolate(method='linear', limit_direction='forward')
df1.fillna(df1.mean(), inplace=True)

normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(df1)
X = pd.DataFrame(training, columns=features)
y = df1.rlf.values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
#cntr1 = Counter (y)
oversample = SMOTE()
X_train_sm, Y_train_sm = oversample.fit_resample(X_train,Y_train)


def run(training_data, test_data, num_runs=100, num_kernels=100):
    results = np.zeros(num_runs)

    Y_training, X_training = training_data[:, 0].astype(np.int), training_data[:, 1:]
    Y_Test, X_Test = test_data[:, 0].astype(np.int), test_data[:, 1:]

    for i in range(num_runs):
        input_length = X_training.shape[1]
        kernels = generate_kernels(input_length, num_kernels)

        # -- transform training ------------------------------------------------

        X_training_transform = apply_kernels(X_training, kernels)

        # -- transform test ----------------------------------------------------

        X_test_transform = apply_kernels(X_Test, kernels)

        # -- training ----------------------------------------------------------

        classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)
        classifier.fit(X_training_transform, Y_training)

        # -- test --------------------------------------------------------------

        results[i] = classifier.score(X_test_transform, Y_Test)

    return results


results = run(X_train_sm,Y_train_sm,100,100)

print(results)


#cntr2 = Counter(Y_train_sm)

#X_train_sm.to_csv("generated2.csv")
# print(X_train_sm.head())
# print(Y_train_sm)
#print(cntr1,cntr2)
# classifier = LinearSVC(random_state=0)
# classifier.fit(X_train_sm, Y_train_sm)
# Y_test_prediction = classifier.predict(X_test)
# conf_matrix = confusion_matrix(Y_test, Y_test_prediction)
# plt.figure()
# sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
# plt.title('SVM Classification - Confusion Matrix')
# print(classification_report(Y_test, Y_test_prediction, target_names=["regular", "rlf"]))
