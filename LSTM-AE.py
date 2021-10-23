'''Implementation of Variational Autoencoder Network for dataset reconstructing into normalized form.
 The whole combined dataset is fed into model by spliting batches'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.io.clipboard import copy
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
from torch.nn import Sequential
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
import time_cv
import datetime
from datetime import datetime as dt

# Train_data
data_train_col = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2_h.csv")
data_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2.csv")
target_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_v2.csv")
data_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_v2.csv")
target_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_v2.csv")

dimension = len(list(data_train_col.columns))

# setting custom lookback index

def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # looking at the past data
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)


# def flatten(X):
#     flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
#     for i in range(X.shape[0]):
#         flattened_X[i] = X[i, (X.shape[1] - 1), :]
#         return flattened_X


batch_size = 32
normalizer = preprocessing.StandardScaler()
training = normalizer.fit_transform(data_train)
# training = normalizer.fit(flatten(data_train))
train_tensor = torch.tensor(training)
train_tensor = train_tensor.float()
target_train = target_train.to_numpy()
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=False)

# Test data
testing = normalizer.fit_transform(data_test)
test_tensor = torch.tensor(testing)
test_tensor = test_tensor.float()
test_X = testing
test_Y = target_test.to_numpy()

timesteps = 3559
n_features = dimension
batch_size = batch_size
lr = 1e-3
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup

lstm_autoencoder = Sequential()


# Encoder
class Encoder(nn.Module):
    def __init__(self, timesteps, n_features, batch_size):
        super(Encoder, self).__init__()
        self.enc1 = lstm_autoencoder.add(
            LSTM(256, activation='relu', input_shape=(timesteps, n_features, batch_size), return_sequences=True))
        self.enc2 = lstm_autoencoder.add(
            LSTM(128, activation='relu', input_shape=(timesteps, n_features, batch_size), return_sequences=True))
        self.enc3 = lstm_autoencoder.add(
            LSTM(64, activation='relu', input_shape=(timesteps, n_features, batch_size), return_sequences=True))
        self.enc4 = lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=False))
        self.enc5 = lstm_autoencoder.add(RepeatVector(timesteps))

    def forward(self, x):
        x = x.reshape((batch_size, self.timesteps, self.n_features))
        x, (_, _) = self.enc1(x)
        x, (_, _) = self.enc2(x)
        x, (_, _) = self.enc3(x)
        x, (_, _) = self.enc4(x)
        x, (hidden_n, _) = self.enc5(x)
        return hidden_n.reshape((self.n_features, self.batch_size))


# Decoder

class Decoder(nn.Module):
    def __init__(self, timesteps, n_features=dimension, compressed_dim=11):
        super(Decoder, self).__init__()

        self.dec1 = lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
        self.dec2 = lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
        self.dec3 = lstm_autoencoder.add(LSTM(128, activation='relu', return_sequences=True))
        self.dec4 = lstm_autoencoder.add(LSTM(256, activation='relu', return_sequences=True))
        self.dec5 = lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.timesteps, self.n_features)
        x = x.reshape((self.n_features, self.timesteps, self.compressed_dim))
        x = x.reshape((self.timesteps, self.hidden_dim))
        return self.output_layer(x)


class LSTM_AE(nn.Module):
    def __init__(self, timesteps, n_features, batch_size=32, compressed_dim=11):
        super().__init__()
        self.encoder = Encoder(timesteps, n_features, batch_size).to(device)
        self.decoder = Decoder(timesteps, n_features, compressed_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Defining model

model = LSTM_AE(train_tensor, data_train.size, dimension)
model = model.to(device)


# Training
def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history


perform_training = train_model(model, train_tensor, test_tensor, n_epochs=50)

all_dataset = zip(train_tensor, test_tensor)


# capturing loss

def get_loss(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset.key():
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            losses.append(loss.item())
    return losses


final_loss = get_loss(perform_training, all_dataset)
loss_df = pd.DataFrame("train_loss", "val_loss")
loss_df["train_loss"] = final_loss[0]
loss_df["val_loss"] = final_loss[1]

# Loss Plotiing

plt.plot(loss_df['loss'], linewidth=2, label='Train')
plt.plot(loss_df['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('LSTM-AE Loss plot')
plt.ylabel('Reconstruction Loss')
plt.xlabel('Epoch')
plt.show()

# Loss Classification

fpr, tpr, thresholds = roc_curve(y_true=test_Y.astype(int), y_score=final_loss[1], pos_label=1)
ranked_thresholds = sorted(list(zip(np.abs(1.5 * tpr - fpr), thresholds, tpr, fpr)), key=lambda i: i[0], reverse=True)
_, failure_threshold, threshold_tpr, threshold_fpr = ranked_thresholds[0]
print(f"Selected failure Threshold: {failure_threshold}")
print("Theshold yields TPR: {:.4f}, FPR: {:.4f}".format(threshold_tpr, threshold_fpr))

auc = roc_auc_score(y_true=test_Y.astype(int), y_score=final_loss[1])
print("AUC: {:.4f}".format(auc))

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], linestyle="--")  # plot baseline curve
plt.plot(fpr, tpr, marker=".",
         label="Failure Threshold:{:.6f}\nTPR: {:.4f}, FPR:{:.4f}".format(failure_threshold, threshold_tpr,
                                                                          threshold_fpr))
plt.axhline(y=threshold_tpr, color='darkgreen', lw=0.8, ls='--')
plt.axvline(x=threshold_fpr, color='darkgreen', lw=0.8, ls='--')
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc="lower right")
plt.show()
test_results = test_Y.astype(bool)
test_results['loss'] = pd.Series(final_loss[1])
test_results['is_failed'] = test_results['loss'] > failure_threshold

conf_matrix = confusion_matrix(test_results['loss'], test_results['is_failed'])
plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title('Failure Threshold Classification - Confusion Matrix')
print(classification_report(test_results['loss'], test_results['is_failed'], target_names=["regular", "rlf"]))
