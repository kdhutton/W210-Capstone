import torch
from torchvision import datasets, transforms
from pycocotools.coco import COCO
import json
import io
import boto3
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from PIL import Image
from torchvision.transforms import transforms, RandAugment
import torch
import tarfile
import os
import getpass
import s3fs
import json
from urllib.parse import urlparse

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

class DataSet(Dataset):
    def __init__(self, ann_files, augs, img_size, dataset):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ]
        )
        self.anns = []
        self.s3_client = boto3.client('s3') 
        self.load_anns()
        print(self.augment)

        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )

    def extract_label_from_filename(self, file_name):
        # Split the path and extract the part with '--'
        parts = file_name.split('/')
        for part in parts:
            if '--' in part:
                # Extract the numeric part before '--'
                label = part.split('--')[0]
                if label.isdigit():
                    return int(label)  # Return as an integer
        raise ValueError(f"Label not found in file name: {file_name}")

        

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs: # need to review RandAugment()
            t.append(RandAugment())
            # t.append(transforms.RandomApply([
            #     transforms.RandomRotation(degrees=10),
            #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            #     transforms.RandomPerspective(distortion_scale=0.05)
            # ], p=0.5))

        t.append(transforms.Resize((img_size, img_size)))
    
        return transforms.Compose(t)

    def load_anns(self):
        s3_client = boto3.client('s3')
        self.anns = []
        for ann_file in self.ann_files:
            bucket, key = self.parse_s3_path(ann_file)
            response = s3_client.get_object(Bucket=bucket, Key=key)
            json_data = json.loads(response['Body'].read())
            for image in json_data['images']:
                file_name = image['file_name']
                label = self.extract_label_from_filename(file_name)  # Use the new method
                for target in image['targets']:
                    ann = {
                        'img_path': f's3://210bucket/WIDER/{file_name}',
                        'bbox': target['bbox'],
                        'label': label,  
                        'target': target['attribute']
                    }
                    self.anns.append(ann)
        print(f"Loaded annotations: {len(self.anns)}")


    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
    
        if not isinstance(ann, dict) or "img_path" not in ann:
            raise ValueError(f"Annotation at index {idx} is not a dictionary with an 'img_path' key: {ann}")
    
        bucket, key = self.parse_s3_path(ann["img_path"])
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            img = Image.open(io.BytesIO(response['Body'].read())).convert("RGB")
        except self.s3_client.exceptions.NoSuchKey:
            print(f"File not found: s3://{bucket}/{key}")
            img = self.get_placeholder_image()
            return None 
    
        x, y, w, h = ann['bbox']
        img = img.crop((x, y, x + w, y + h))
        img = self.augment(img)
        img = self.transform(img)
    
        label = torch.tensor(ann['label'], dtype=torch.long)    
        target = torch.tensor(ann['target'], dtype=torch.float32)
    
        message = {
            "label": label,  
            "target": target,  
            "img": img 
        }
    
        return message


    def get_placeholder_image(self):
        return Image.new('RGB', (256, 256), color = 'gray')

    @staticmethod
    def parse_s3_path(s3_path):
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        s3_path = s3_path[5:]
        bucket, key = s3_path.split('/', 1)
        return bucket, key
        


def load_wider(batch_size=64, subset_size=None):

    train_augs = ['randomflip', 'ColorJitter', 'resizedcrop', 'RandAugment']
    test_augs = []  
    img_size = 256 
    
    train_dataset = DataSet(
        ann_files=['s3://210bucket/wider_attribute_annotation/wider_attribute_trainval.json'],  
        augs=train_augs,
        img_size=img_size,
        dataset='wider'
    )
    
    test_dataset = DataSet(
        ann_files=['s3://210bucket/wider_attribute_annotation/wider_attribute_test.json'], 
        augs=test_augs,
        img_size=img_size,
        dataset='wider'
    )
    
    if subset_size is not None:
        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), subset_size, replace=False)
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


