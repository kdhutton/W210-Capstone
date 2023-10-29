import torch
from torchvision import datasets, transforms
from pycocotools.coco import COCO

def load_cifar10(batch_size=64):
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_cifar100(batch_size=64):
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_imagenet(train_path, test_path, batch_size=64):
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
    
def load_prof(train_path, test_path, batch_size=64):
    
    # Define data transformations (you can customize this based on your needs)
    transform = transforms.Compose([
        transforms.Resize((226, 226)),  # Resize the images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])
    
    # Create the ImageFolder dataset
    traindataset = datasets.ImageFolder(root=train_path, transform=transform)
    testdataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    
    # Create a DataLoader to load the data
    # batch_size = 32  # You can adjust this based on your hardware and requirements
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, test_loader

def load_coco(data_dir, batch_size=64):
    # Load Data - COCO
    # Adjust mean and std values as appropriate
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # check the data directory
    train_set = datasets.CocoDetection(root=f'{data_dir}/train2017', 
                                       annFile=f'{data_dir}/annotations/instances_train2017.json', 
                                       transform=transform)
    test_set = datasets.CocoDetection(root=f'{data_dir}/val2017', 
                                     annFile=f'{data_dir}/annotations/instances_val2017.json', 
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


