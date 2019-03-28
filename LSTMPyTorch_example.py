#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:35:04 2019

Pre-packaged LSTM from Pytorch

@author: mb540
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Global
torch.set_default_dtype(torch.float64)

# Network
learning_rate = 1e-3
epochs = 500
input_size = 1
output_size = 1
sequence_length = 1
hidden_size = 64

# Training
num_datapoints = 100
test_size = 0.2
num_train = int((1-test_size) * num_datapoints)

#####################
# Read data from file
######################
df = pd.read_csv('./Pre-processed_data.csv')

### Prepare data
df_output = df.drop(['LMotor', 'RMotor'], 1)
# One time series at a time (testing)
df = df['XCoord']
df_output = df

# Give a window of lenght sequence_length
X = torch.zeros(sequence_length,0)
for i in range(num_datapoints):
    X = torch.cat((X, torch.tensor(df[i:i+sequence_length].values).view([-1,1])),1)

X_train = X[:, :num_train]
X_test = X[:, num_train:num_datapoints]

y_train = torch.tensor(df_output[sequence_length:sequence_length+num_train].values)
y_test = torch.tensor(df_output[sequence_length+num_train:num_datapoints+sequence_length].values)

# Format for lstm method
X_train = X_train.view([sequence_length, -1, input_size])
X_test = X_test.view([sequence_length, -1, input_size])


lstm = torch.nn.LSTM(input_size, hidden_size)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

hist = np.zeros(epochs)

# Stick to Pytorch language in the future? batch_size = num_train

for t in range(epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # lstm.hidden = lstm.init_hidden()
    
    # Forward pass
    lstm_out, lstm_hidden = lstm(X_train)

    # Reduce dimensions to match output_size (to check)
    linear = torch.nn.Linear(hidden_size, output_size)

    # print(lstm_out.size())
    # print(lstm_out[-1].view(num_train, -1).size())

    y_pred = linear(lstm_out[-1].view(num_train, -1))

    print(y_pred.view(-1), -y_train)

    loss = loss_fn(y_pred.view(-1), -y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

# print(lstm_out.size())
# print(y_pred.size())
# print(y_train.size())

#####################
# Plot preds and performance
#####################
plt.figure()
# plt.subplot(3,2,1)
plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
# plt.subplot(3,2,2)
# plt.plot(lstm_out[:,0,1].detach().numpy(), label="Preds")
# plt.plot(y_train[:,1].detach().numpy(), label="Data")
# plt.subplot(3,2,3)
# plt.plot(lstm_out[:,0,2].detach().numpy(), label="Preds")
# plt.plot(y_train[:,2].detach().numpy(), label="Data")
# plt.subplot(3,2,4)
# plt.plot(lstm_out[:,0,3].detach().numpy(), label="Preds")
# plt.plot(y_train[:,3].detach().numpy(), label="Data")
# plt.subplot(3,2,5)
# plt.plot(lstm_out[:,0,4].detach().numpy(), label="Preds")
# # plt.plot(y_train[:,4].detach().numpy(), label="Data")
# plt.subplot(3,2,6)
# plt.plot(lstm_out[:,0,5].detach().numpy(), label="Preds")
# # plt.plot(y_train[:,5].detach().numpy(), label="Data")
# plt.legend()

plt.figure()
# plt.subplot(3,2,1)
plt.plot(y_train.detach().numpy(), label="Data")

plt.figure()
# plt.subplot(3,2,1)
plt.plot(y_pred.detach().numpy(), label="Preds")

plt.figure()
plt.plot(hist, label="Training loss")
plt.legend()
plt.show()