import torch
from torchvision import datasets
import torchvision.transforms as T
from torchvision import datasets, transforms


def IDENPROF(name='idenprof', valid_size=None):

    # Define data transformations (you can customize this based on your needs)
    transform = transforms.Compose([
        transforms.Resize((226, 226)),  # Resize the images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    # Load IdenProf dataset
    train_path = '/home/ubuntu/W210-Capstone/competition_models/MB_Knowledge-Distillation-via-ND/CIFAR/idenprof/train'
    test_path = '/home/ubuntu/W210-Capstone/competition_models/MB_Knowledge-Distillation-via-ND/CIFAR/idenprof/test'
    

    # Create the ImageFolder dataset
    train_data = datasets.ImageFolder(root=train_path, transform=transform)
    test_data = datasets.ImageFolder(root=test_path, transform=transform)
    num_class = 10

    
    
    return train_data, test_data, num_class