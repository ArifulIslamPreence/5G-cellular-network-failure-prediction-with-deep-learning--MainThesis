import numpy as np
import pandas as pd
from sklearn import preprocessing


class spliting_window(object):

    def __init__(self, initial=1, horizon=1, period=1):
        self.initial = initial  # original Data size
        self.horizon = horizon  # test data size
        self.period = period    # interval period in spliting

    def split(self, data):
        self.data = data
        self.counter = 0  # for us to iterate and track later

        data_length = data.shape[0]  # rows
        data_index = list(np.arange(data_length))

        output_train = []
        output_test = []
        # append initial
        output_train.append(list(np.arange(self.initial)))
        progress = [x for x in data_index if x not in list(np.arange(self.initial))]  # indexes left to append to train
        output_test.append([x for x in data_index if x not in output_train[self.counter]][:self.horizon])
        # clip initial indexes from progress since that is what we are left

        while len(progress) != 0:
            temp = progress[:self.period]
            to_add = output_train[self.counter] + temp
            # update the train index
            output_train.append(to_add)
            # increment counter
            self.counter += 1

            # update the test index
            to_add_test = [x for x in data_index if x not in output_train[self.counter]][:self.horizon]
            output_test.append(to_add_test)

            # update progress
            progress = [x for x in data_index if x not in output_train[self.counter]]

        # clip the last element of output_train and output_test
        output_train = output_train[:-1]
        output_test = output_test[:-1]

        # mimic sklearn output
        index_output = [(train, test) for train, test in zip(output_train, output_test)]

        return index_output


df1 = pd.read_csv("output_dataset/new_combined.csv", index_col=0, low_memory=False)
features = df1.columns
df1 = df1.interpolate(method='linear', limit_direction='forward')
df1.fillna(df1.mean(), inplace=True)

normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(df1)
X = pd.DataFrame(training, columns=features)
y = df1.rlf.values

tscv = spliting_window(initial = 30, horizon = 16,period = 1)
for train_index, test_index in tscv.split(X):
    print(train_index)
    print(test_index)
