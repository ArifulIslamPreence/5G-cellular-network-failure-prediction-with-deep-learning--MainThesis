'''Implementation of Variational Autoencoder Network for dataset reconstructing into normalized form.
 The whole combined dataset is fed into model by spliting batches'''

import csv
import numpy as np
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
import time_cv
import datetime
from datetime import datetime as dt

batch_size = 1

# Train_data
data_train_col = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2_h.csv")
data_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2.csv")
target_train = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_train_ol_v2.csv")
data_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_v2.csv")
target_test = pd.read_csv("../output_dataset/cv_train_test/outlierfree/y_test_ol_v2.csv")

normalizer = preprocessing.StandardScaler()
# training = normalizer.fit_transform(x_trains)
training = normalizer.fit_transform(data_train)
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
#
#
#
dimension = len(list(data_train_col.columns))

lr = 1e-3
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encoder Structure

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=dimension):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))


# Decoder Structure

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        # x, (hidden_n, cell_n) = self.rnn1(x)
        # x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define Model

model = RecurrentAutoencoder(train_tensor, data_train.size, len(dimension))
model = model.to(device)

# Training Model

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

get_training_loss = training(model,data_train,)

# Generate Prediction Score

def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


prediction_result = predict(model,data_train,data_test)

fpr, tpr, thresholds = roc_curve(y_true=test_Y.astype(int), y_score=test_loss, pos_label=1)
ranked_thresholds = sorted(list(zip(np.abs(1.5 * tpr - fpr), thresholds, tpr, fpr)), key=lambda i: i[0], reverse=True)
_, failure_threshold, threshold_tpr, threshold_fpr = ranked_thresholds[0]
print(f"Selected failure Threshold: {failure_threshold}")
print("Theshold yields TPR: {:.4f}, FPR: {:.4f}".format(threshold_tpr, threshold_fpr))

auc = roc_auc_score(y_true=test_Y.astype(int), y_score=test_loss)
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
test_results['loss'] = pd.Series(test_loss)
test_results['is_failed'] = test_results['loss'] > failure_threshold

conf_matrix = confusion_matrix(test_results['loss'], test_results['is_failed'])
plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title('Failure Threshold Classification - Confusion Matrix')
print(classification_report(test_results['loss'], test_results['is_failed'], target_names=["regular", "rlf"]))
