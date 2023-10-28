import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_features, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features * 15 * 15, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.conv1.out_channels * 15 * 15)
        x = self.fc1(x)
        return x

class TeacherModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_features, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features * 15 * 15, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.conv1.out_channels * 15 * 15)
        x = self.fc1(x)
        return x
