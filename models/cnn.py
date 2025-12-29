import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) #1x28x28 -> 6x24x24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #6x24x24 -> 6x12x12                                         
        self.conv2 = nn.Conv2d(6, 16, 5) #6x12x12 -> 16x8x8
        # After pooling: 16x4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)