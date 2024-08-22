import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(3, 1), stride=(2, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output from convolutional layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
