import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer (2x2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # Fully connected -> ReLU
        x = torch.relu(self.fc2(x))  # Fully connected -> ReLU
        x = self.fc3(x)  # Output layer
        return x