import torch.nn as nn
import torch.nn.functional as F

# Default models in the base model notebook
class StudentModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_features, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(in_features * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, self.conv1.out_channels * 4 * 4)
        x = self.fc1(x)
        return x


class TeacherModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(3, in_features, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(in_features * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, self.conv1.out_channels * 4 * 4)
        x = self.fc1(x)
        return x

### Classical Knowledge Distillation
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

### Retional Knowledge Distillation
class CustomResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True, num_classes=10):
        super(CustomResNet18, self).__init__()

        # Get the pretrained model
        pretrained_model = torchvision.models.resnet18(pretrained=pretrained)
        
        # Copy layers from the pretrained model
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained_model, module_name))

        # Add an adaptive average pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a final fully connected layer
        self.fc = nn.Linear(self.output_size, num_classes)

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        x = self.avgpool(b4)

        # If get_ha flag is True, return intermediate activations
        if get_ha:
            return b1, b2, b3, b4, x

        # Use the adaptive pooling layer
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Final fully connected layer
        x = self.fc(x)
        return x