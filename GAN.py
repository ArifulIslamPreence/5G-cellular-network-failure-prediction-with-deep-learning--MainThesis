'''
Implementation of Generative Adverserial Network for dataset balancing.
 The whole combined dataset is fed into model by spliting batches
'''
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_train_ol_v2.csv", low_memory=False, header=None)
df2 = pd.read_csv("../output_dataset/cv_train_test/outlierfree/x_test_ol_v2.csv", low_memory=False, header=None)

train, test = df1, df2

dimension = int(len(train.columns))
data_len = int(len(df1))
batchSize = 100

# Train_data
normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(train)
train_tensor = torch.tensor(training.astype(np.float32))

# trainloader needs to filled with the input tensor and input label tensor tuple (data,label) pair
Train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batchSize, shuffle=False)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# GAN implementation

# NN for discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = dimension
        output_dim = 1
        self.label_embedding = nn.Embedding(1, 1)
        self.model = nn.Sequential(

            # setting 3 layers
            nn.Linear(in_features=input_dim, out_features=int(dimension / 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=int(dimension / 2), out_features=int(dimension / 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 8)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=int(dimension / 8), out_features=output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = x.view(dimension)
        # c = self.label_embedding(labels)
        # x = torch.cat([x, c], 1)
        output = self.model(x)
        return output


# NN for fake data generation
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 100
        output_dim = dimension
        self.label_embedding = nn.Embedding(1, 1)
        self.model = nn.Sequential(

            # 3 layers
            nn.Linear(input_dim, out_features=int(dimension / 8)),
            nn.ReLU(),
            nn.Linear(in_features=int(dimension / 8), out_features=int(dimension / 4)),
            nn.ReLU(),
            nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(dimension / 2), out_features=int(output_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        # c = self.label_embedding(labels)
        # x = torch.cat([x, c], 1)
        output = self.model(x)
        return output


gen = Generator().to(device=device)
disc = Discriminator().to(device=device)

# Model Training

lr = 3e-4
num_epoch = 100
loss_function = nn.BCEWithLogitsLoss()

# Model Optimizer

optimizer_discriminator = torch.optim.Adam(disc.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(gen.parameters(), lr=lr)


# Start trainings

def training_gan(generator, discriminator, num_epochs, Batch_size, train_loader):
    for epoch in range(num_epochs):
        lossGen = []
        lossDis = []
        for n, data in enumerate(train_loader):

            real_samples = data
            real_samples_labels = torch.ones((Batch_size, 2)).to(
                device=device
            )
            latent_space_sample = torch.randn((Batch_size, 2))
            generated_sample = generator(latent_space_sample)
            generated_samples_label = torch.zeros(
                (Batch_size, 1))  # create labels with the value 0 for the real samples
            all_samples = torch.cat(
                (real_samples, generated_sample))  # problem here because of the sample size are different
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_label))
            # torch.reshape(all_samples,(200,1))

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator,
                                               all_samples_labels)  # calculate the loss function using the output
            # from the model
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_sample = torch.randn((Batch_size, 1))

            # Training the generator
            generator.zero_grad()
            generated_sample = generator(latent_space_sample)
            output_discriminator_generated = discriminator(
                generated_sample)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)  # loss function
            loss_generator.backward()
            optimizer_generator.step()
            lossGen.append(loss_generator)
            lossDis.append(loss_discriminator)
            # Show loss for each 5 epoch
            if epoch % 10 == 0 and n == Batch_size - 1:
                print(f"Epoch: {epoch + 1} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch + 1} Loss G.: {loss_generator}")
            return lossGen, lossGen


getloss_gen, getloss_dis = training_gan(gen, disc, num_epoch, batchSize, Train_loader)

_, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("Generator loss value ")
ax.set_title(' Generator Loss graph')
ax.plot(getloss_gen)
#
_, bx = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("Discriminator loss value ")
bx.set_title('Discriminator  Loss graph')
bx.plot(getloss_dis)
#
#
# _, ax = plt.subplots(1, 1, figsize=(15, 10))
# plt.xlabel("epochs")
# plt.ylabel("New loss value ")
# ax.set_title(' New Loss graph')
# ax.plot(generated_samples)
#
# Calculate reconstruction loss for test partition (30% real data)
# test_loss = []
# generator.eval()
# discriminator.eval()
# test_tensor = torch.tensor(test_X.values.astype(np.float32))
#
# with torch.no_grad():
#     for i in range(len(test_X)):
#         input = test_tensor[i].to(device=device)
#         output = Generator(input).to(device=device)
#         loss = loss_function(output, input).to(device=device)
#         test_loss.append(loss.item())

latent_space_samples = torch.randn(data_len, dimension).to(device=device)
generated_samples = gen(latent_space_samples)
generated_samples.to_csv("generated2.csv")
