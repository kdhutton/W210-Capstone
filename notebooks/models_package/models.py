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

### Classical Knowledge Distillatino
class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        
        self.layer1 = nn.Linear(3072, 1200)
        self.layer2 = nn.Linear(1200, 1200)
        self.layer3 = nn.Linear(1200, 100)
        
        self.dropout_20 = nn.Dropout(0.2)
        self.dropout_50 = nn.Dropout(0.5)

    def forward(self, x):
        # Check input shape and resize if necessary
        if x.shape[-3:] != (3, 32, 32):
            x = x[:, :3, :32, :32]
        
        x = x.contiguous().view(x.size(0), -1)  # Flatten the input
        x = self.dropout_20(x)
        x = F.relu(self.layer1(x))
        x = self.dropout_50(x)
        x = F.relu(self.layer2(x))
        x = self.dropout_50(x)
        return self.layer3(x)

class Student(nn.Module):
    def __init__(self, use_dropout=False):
        super(Student, self).__init__()
        self.use_dropout = use_dropout
        
        self.layer1 = nn.Linear(3072, 800)
        self.layer2 = nn.Linear(800, 800)
        self.layer3 = nn.Linear(800, 100)
        
        self.dropout_20 = nn.Dropout(0.2)
        self.dropout_50 = nn.Dropout(0.5)

    def forward(self, x):
        # Check input shape and resize if necessary
        if x.shape[-3:] != (3, 32, 32):
            x = x[:, :3, :32, :32]
            
        x = x.contiguous().view(x.size(0), -1)  # Flatten the input
        if self.use_dropout:
            x = self.dropout_20(x)
        x = F.relu(self.layer1(x))
        if self.use_dropout:
            x = self.dropout_50(x)
        x = F.relu(self.layer2(x))
        if self.use_dropout:
            x = self.dropout_50(x)
        return self.layer3(x)