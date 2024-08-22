import argparse
import pickle
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

## Defining the LSTM model architecture


class MovieLSTM(nn.Module):
    def __init__(self, config):
        super(MovieLSTM, self).__init__()
        self.num_neurons = config.num_frame * 2
        self.num_characters = config.num_labels
        hidden_size = 128
        fc_feature2 = 128
        # hidden_size = 64
        num_layers = 2
        dropout = 0
        self.lstm = nn.LSTM(
            self.num_neurons, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Sequential(nn.Linear(hidden_size, fc_feature2))
        self.fc2 = nn.Sequential(nn.Linear(fc_feature2, self.num_characters))
        self.fc_normed = nn.BatchNorm1d(128)
        self.softmax = nn.LogSoftmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        lstm_output, _ = self.lstm(inputs, None)
        # lstm_output1 = self.dropout_layer(lstm_output)
        output = self.relu(self.fc_normed(self.fc(lstm_output[:, -1, :])))
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output


class MovieLSTM_reg(nn.Module):
    def __init__(self, data_dir, tag_dir):
        super().__init__()
        self.num_neurons = np.load(data_dir).shape[1]
        tag_mat = np.load(tag_dir, allow_pickle=True)
        classnames, indices = np.unique(tag_mat, return_inverse=True)
        self.num_tag = self.num_neurons * len(classnames)
        # self.__dict__.update(kwargs)
        hidden_size = 512
        num_layers = 2
        dropout = 0.5
        self.lstm = nn.LSTM(
            self.num_neurons, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        # self.fc = nn.Sequential(nn.Linear(hidden_size, num_cha)
        self.fc1 = nn.Linear(in_features=(hidden_size + self.num_tag), out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        self.fc1_dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=num_characters * 3)
        self.fc2_normed = nn.BatchNorm1d(num_characters * 3)
        self.softmax = nn.LogSoftmax(dim=2)
        self.relu = nn.LeakyReLU()

    def forward(self, inputs, region_tag):
        inputs = inputs.float()
        lstm_output, _ = self.lstm(inputs, None)
        lstm_output1 = self.dropout_layer(lstm_output)
        output_lstm = lstm_output1[:, -1, :]
        # output = output.reshape(-1, num_characters, 3)
        # output = self.softmax(output)
        # print(output_lstm.shape, region_tag.shape, self.num_tag)

        # batch = torch.cat((output_lstm.view(-1, self.num_flat_features(output_lstm)), region_tag), 1)
        batch = torch.cat((output_lstm, region_tag), 1)

        # print("batch.shape", batch.shape)
        # print("batch_2.shape", self.fc1(batch).shape)
        # batch_2 = self.fc1_normed(self.fc1(batch))
        # print(batch_2.shape, self.fc1(batch).shape)

        batch = self.fc1_dropout(self.relu(self.fc1_normed(self.fc1(batch))))
        batch = self.fc2(batch)
        batch = batch.reshape(-1, num_characters, 3)
        # batch = func.softmax(batch, dim=2)
        batch = self.softmax(batch)
        # print("batch.shape", batch.shape)

        return batch

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
