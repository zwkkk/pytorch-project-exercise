## build CNN
from torch import nn


## build CNN
class Net(nn.Module):
    # def __init__(self,num_classes=10):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.relu4 = nn.ReLU(True)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.bn1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.bn2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.bn3(output)
        output = self.pool3(output)

        output = output.view(-1, 128 * 16 * 16)
        output = self.fc1(output)
        output = self.relu4(output)
        output = self.fc2(output)

        return output