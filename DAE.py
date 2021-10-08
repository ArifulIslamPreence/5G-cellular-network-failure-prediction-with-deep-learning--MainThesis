#
#
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#         # encoder
#         self.enc1 = nn.Linear(in_features=dimension, out_features=int(dimension / 2))
#         self.enc2 = nn.Linear(in_features=int(dimension / 2), out_features=int(dimension / 4))
#         self.enc3 = nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 8))
#         self.enc4 = nn.Linear(in_features=int(dimension / 8), out_features=int(dimension / 16))
#
#         # decoder
#         self.dec1 = nn.Linear(in_features=int(dimension / 16), out_features=int(dimension / 8))
#         self.dec2 = nn.Linear(in_features=int(dimension / 8), out_features=int(dimension / 4))
#         self.dec3 = nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 2))
#         self.dec4 = nn.Linear(in_features=int(dimension / 2), out_features=dimension)
#
#     def forward(self, x):
#         # x = F.relu(self.enc1(x))
#         # x = F.relu(self.enc2(x))
#         # x = F.relu(self.enc3(x))
#         #
#         # x = F.relu(self.dec1(x))
#         # x = F.relu(self.dec2(x))
#         # x = F.relu(self.dec3(x))
#
#         # sigmoid activation
#         x = torch.sigmoid(self.enc1(x))
#         x = torch.sigmoid(self.enc2(x))
#         x = torch.sigmoid(self.enc3(x))
#         x = torch.sigmoid(self.enc4(x))
#         # x = F.relu(self.enc4(x))
#
#         x = torch.sigmoid(self.dec1(x))
#         x = torch.sigmoid(self.dec2(x))
#         x = torch.sigmoid(self.dec3(x))
#         x = torch.sigmoid(self.dec4(x))
#         return x
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = AutoEncoder()
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
#
# loss_function = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss()  #MSELoss too
# get_loss = list()
#
#
# def training_ae(net, trainloader, epochs):
#     train_loss = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for data in train_loader:
#             input_data = data.to(device=device)
#             optimizer.zero_grad()
#             output = net(input_data).to(device=device)  # output is the reconstruced x
#             loss = loss_function(output, input_data).to(device=device)  # input_data should be the target variable
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         loss = running_loss / len(trainloader)
#         train_loss.append(loss)
#         outputs.append(output)
#
#         if epoch % 5 == 0:
#             print('Epoch {} of {}, Train Loss: {:.3f}'.format(
#                 epoch + 1, num_epochs, loss))
#     return train_loss
#
#
# get_loss_train = training_ae(net, train_loader, num_epochs)
#
# _, ax = plt.subplots(1, 1, figsize=(15, 10))
# plt.xlabel("epochs")
# plt.ylabel("loss value ")
# ax.set_title('Loss graph')
# ax.plot(get_loss_train)
# plt.show()
# test_loss = []
# net.eval()
#
# with torch.no_grad():
#     for i in range(len(test_X)):
#         input = test_tensor[i].to(device=device)
#         output = net(input).to(device=device)
#         loss = loss_function(output, input).to(device=device)
#         test_loss.append(loss.item())