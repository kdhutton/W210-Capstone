import torch
from torchvision import datasets, transforms

def load_cifar10():
    # Load Data - CIFAR10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def load_cifar100():
    # Load Data - CIFAR100
    # Adjust mean and std values as appropriate
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def load_imagenet(train_path, test_path):
    # Load Data - ImageNet
    # Adjust mean and std values as appropriate
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_set = datasets.ImageFolder(root=train_path, transform=transform)
    test_set = datasets.ImageFolder(root=test_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, test_loader
    


