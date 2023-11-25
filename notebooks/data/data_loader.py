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
import torchvision
# from torchvision.transforms import transforms # this may mess up other dataloaders, for wider
import torchvision.transforms as transforms
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


# class DataSet(Dataset):
#     def __init__(self, ann_files, augs, img_size, dataset, undersampe=False):
#         self.dataset = dataset
#         self.ann_files = ann_files
#         self.augment = self.augs_function(augs, img_size)
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0, 0, 0], 
#                                      std=[1, 1, 1])
#             ]
#         )
#         self.anns = []
#         self.s3_client = boto3.client('s3') 
#         self.load_anns()
#         print(self.augment)

#         if self.dataset == "wider":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5, 0.5, 0.5], 
#                                          std=[0.5, 0.5, 0.5])
#                 ]
#             )

#     def extract_label_from_filename(self, file_name):
#         # Split the path and extract the part with '--'
#         parts = file_name.split('/')
#         for part in parts:
#             if '--' in part:
#                 # Extract the numeric part before '--'
#                 label = part.split('--')[0]
#                 if label.isdigit():
#                     return int(label)  # Return as an integer
#         raise ValueError(f"Label not found in file name: {file_name}")

        

#     def augs_function(self, augs, img_size):            
#         t = []
#         if 'randomflip' in augs:
#             t.append(transforms.RandomHorizontalFlip())
#         if 'ColorJitter' in augs:
#             t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
#         if 'resizedcrop' in augs:
#             t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
#         if 'RandAugment' in augs: # need to review RandAugment()
#             t.append(RandAugment())
#             # t.append(transforms.RandomApply([
#             #     transforms.RandomRotation(degrees=10),
#             #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             #     transforms.RandomPerspective(distortion_scale=0.05)
#             # ], p=0.5))

#         t.append(transforms.Resize((img_size, img_size)))
    
#         return transforms.Compose(t)

#     def load_anns(self):
#         s3_client = boto3.client('s3')
#         self.anns = []
#         for ann_file in self.ann_files:
#             bucket, key = self.parse_s3_path(ann_file)
#             response = s3_client.get_object(Bucket=bucket, Key=key)
#             json_data = json.loads(response['Body'].read())
#             for image in json_data['images']:
#                 file_name = image['file_name']
#                 label = self.extract_label_from_filename(file_name)  # Use the new method
#                 for target in image['targets']:
#                     ann = {
#                         'img_path': f's3://210bucket/WIDER/{file_name}',
#                         'bbox': target['bbox'],
#                         'label': label,  
#                         'target': target['attribute']
#                     }
#                     self.anns.append(ann)
#         print(f"Loaded annotations: {len(self.anns)}")


#     def undersample_anns(self):
#         # Shuffle annotations before undersampling
#         random.shuffle(self.anns)

#         # Count the instances per class
#         class_counts = {}
#         for ann in self.anns:
#             label = self.extract_label(ann['img_path'])  # Assuming this method returns the class label
#             class_counts[label] = class_counts.get(label, 0) + 1

#         # Find the minimum class count
#         min_count = min(class_counts.values())

#         # Perform undersampling
#         undersampled_anns = []
#         current_counts = {label: 0 for label in class_counts}
#         for ann in self.anns:
#             label = self.extract_label(ann['img_path'])
#             if current_counts[label] < min_count:
#                 undersampled_anns.append(ann)
#                 current_counts[label] += 1

#         # Update the annotations to the undersampled list
#         self.anns = undersampled_anns
        
#     def __len__(self):
#         return len(self.anns)

#     def __getitem__(self, idx):
#         idx = idx % len(self)
#         ann = self.anns[idx]
    
#         if not isinstance(ann, dict) or "img_path" not in ann:
#             raise ValueError(f"Annotation at index {idx} is not a dictionary with an 'img_path' key: {ann}")
    
#         bucket, key = self.parse_s3_path(ann["img_path"])
#         try:
#             response = self.s3_client.get_object(Bucket=bucket, Key=key)
#             img = Image.open(io.BytesIO(response['Body'].read())).convert("RGB")
#         except self.s3_client.exceptions.NoSuchKey:
#             print(f"File not found: s3://{bucket}/{key}")
#             img = self.get_placeholder_image()
#             return None 
    
#         x, y, w, h = ann['bbox']
#         img = img.crop((x, y, x + w, y + h))
#         img = self.augment(img)
#         img = self.transform(img)
    
#         label = torch.tensor(ann['label'], dtype=torch.long)    
#         target = torch.tensor(ann['target'], dtype=torch.float32)
    
#         message = {
#             "label": label,  
#             "target": target,  
#             "img": img 
#         }
    
#         return message


#     def get_placeholder_image(self):
#         return Image.new('RGB', (256, 256), color = 'gray')

#     @staticmethod
#     def parse_s3_path(s3_path):
#         if not s3_path.startswith("s3://"):
#             raise ValueError(f"Invalid S3 path: {s3_path}")
#         s3_path = s3_path[5:]
#         bucket, key = s3_path.split('/', 1)
#         return bucket, key
        


# def load_wider(batch_size=64, subset_size=None):

#     train_augs = ['randomflip', 'ColorJitter', 'resizedcrop', 'RandAugment']
#     test_augs = []  
#     img_size = 256 
    
#     train_dataset = DataSet(
#         ann_files=['s3://210bucket/wider_attribute_annotation/wider_attribute_trainval.json'],  
#         augs=train_augs,
#         img_size=img_size,
#         dataset='wider'
#     )
    
#     test_dataset = DataSet(
#         ann_files=['s3://210bucket/wider_attribute_annotation/wider_attribute_test.json'], 
#         augs=test_augs,
#         img_size=img_size,
#         dataset='wider'
#     )
    
#     if subset_size is not None:
#         train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
#         test_indices = np.random.choice(len(test_dataset), subset_size, replace=False)
        
#         train_subset = Subset(train_dataset, train_indices)
#         test_subset = Subset(test_dataset, test_indices)
        
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
#     else:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


# ## LB version below ##
# class DataSet(Dataset):
#     def __init__(self, ann_files, augs, img_size, dataset, undersample=False):
#         # Define the original class labels
#         class_labels = [0, 1, 3, 4, 6, 7, 11, 15, 17, 18, 19, 20, 22, 25, 27, 28, 30, 31, 33, 35, 36, 37, 39, 43, 44, 50, 51, 54, 57, 58]
#         class_labels_new = torch.tensor([i for i in range(len(class_labels))])
        
#         # Create a mapping from old labels to new labels
#         self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(class_labels))}

#         self.dataset = dataset
#         self.ann_files = ann_files
#         self.augment = self.augs_function(augs, img_size)
#         # Initialize transformations directly
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0, 0, 0],
#                                      std=[1, 1, 1])
#             ] 
#         )
#         if self.dataset == "wider":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#                 ] 
#             )        

#         self.anns = []
#         self.load_anns()
#         if undersample: 
#             self.undersample_anns()
#         print(self.augment)

#     def augs_function(self, augs, img_size):            
#         t = []
#         if 'randomflip' in augs:
#             t.append(torchvision.transforms.RandomHorizontalFlip())
#         if 'ColorJitter' in augs:
#             t.append(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
#         if 'resizedcrop' in augs:
#             t.append(torchvision.transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
#         if 'RandAugment' in augs:
#             t.append(torchvision.transforms.RandAugment())

#         t.append(transforms.Resize((img_size, img_size)))

#         return transforms.Compose(t)
    
#     def load_anns(self):
#         self.anns = []
#         for ann_file in self.ann_files:
#             json_data = json.load(open(ann_file, "r"))
#             self.anns += json_data

#     def undersample_anns(self):
#         # Shuffle annotations before undersampling
#         random.shuffle(self.anns)

#         # Count the instances per class
#         class_counts = {}
#         for ann in self.anns:
#             label = self.extract_label(ann['img_path'])  # Assuming this method returns the class label
#             class_counts[label] = class_counts.get(label, 0) + 1

#         # Find the minimum class count
#         min_count = min(class_counts.values())

#         # Perform undersampling
#         undersampled_anns = []
#         current_counts = {label: 0 for label in class_counts}
#         for ann in self.anns:
#             label = self.extract_label(ann['img_path'])
#             if current_counts[label] < min_count:
#                 undersampled_anns.append(ann)
#                 current_counts[label] += 1

#         # Update the annotations to the undersampled list
#         self.anns = undersampled_anns
    
#     def __len__(self):
#         return len(self.anns)

#     def __getitem__(self, idx):
#         # Make sure the index is within bounds
#         idx = idx % len(self)
#         ann = self.anns[idx]
        
#         try:
#             # Attempt to open the image file
#             img = Image.open(f'WIDER/Image/{ann["file_name"]}').convert("RGB")

#             # If this is the wider dataset, proceed with specific processing
#             # x, y, w, h = ann['bbox']
#             # img_area = img.crop([x, y, x+w, y+h])
#             img_area = self.augment(img)
#             img_area = self.transform(img_area)
#             attributes_list = [target['attribute'] for target in ann['targets']]
#             summed_attributes = [max(sum(attribute), 0) for attribute in zip(*attributes_list)]
#             # Extract label from image path
#             img_path = f'WIDER/Image/{ann["file_name"]}'
#             label = self.extract_label(img_path)  # You might need to implement this method
            
#             return {
#                 "label": label,
#                 "target": torch.Tensor(summed_attributes),
#                 "img": img_area
#             }
            

#         except Exception as e:
#             # If any error occurs during the processing of an image, log the error and the index
#             print(f"Error processing image at index {idx}: {e}")
#             # Instead of returning None, raise the exception
#             raise

#     def extract_label(self, img_path):
#         original_label = None
    
#         if "WIDER/Image/train" in img_path:
#             label_str = img_path.split("WIDER/Image/train/")[1].split("/")[0]
#             original_label = int(label_str.split("--")[0])
#         elif "WIDER/Image/test" in img_path:
#             label_str = img_path.split("WIDER/Image/test/")[1].split("/")[0]
#             original_label = int(label_str.split("--")[0])
#         elif "WIDER/Image/val" in img_path:  # Handle validation images
#             label_str = img_path.split("WIDER/Image/val/")[1].split("/")[0]
#             original_label = int(label_str.split("--")[0])
    
#         if original_label is not None:
#             remapped_label = self.label_mapping[original_label]
#             return remapped_label
#         else:
#             raise ValueError(f"Label could not be extracted from path: {img_path}")

class DataSet(Dataset):
    def __init__(self, ann_files, class_labels, augs, img_size, dataset):
        # Define the original class labels
        self.class_labels = class_labels

        # Create a mapping from old labels to new labels
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(self.class_labels))}

        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        # Initialize transformations directly
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
        )
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                ] 
            )        

        self.anns = []
        self.load_anns()
        print(self.augment)

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(transforms.RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            # json_data = json.load(open(ann_file, "r"))
            ann_file = ann_file[0] # need to extract from list format
            with open(ann_file, "r") as file:
                json_data = json.load(file)
                
            self.anns += json_data['images']

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        # Make sure the index is within bounds
        idx = idx % len(self)
        ann = self.anns[idx]
        
        try:
            # Attempt to open the image file
            img = Image.open(f'data/WIDER/Image/{ann["file_name"]}').convert("RGB")

            # If this is the wider dataset, proceed with specific processing
            # x, y, w, h = ann['bbox']
            # img_area = img.crop([x, y, x+w, y+h])
            img_area = self.augment(img)
            img_area = self.transform(img_area)
            attributes_list = [target['attribute'] for target in ann['targets']]
            num_people = len(attributes_list)
            attributes_distribution = [max(sum(attribute), 0)/num_people for attribute in zip(*attributes_list)]
            # Extract label from image path
            img_path = f'data/WIDER/Image/{ann["file_name"]}'
            label = self.extract_label(img_path)  # You might need to implement this method
            
            return {
                "label": label,
                "target": torch.tensor([attributes_distribution[0]], dtype=torch.float32),
                "img": img_area
            }
            

        except Exception as e:
            # If any error occurs during the processing of an image, log the error and the index
            print(f"Error processing image at index {idx}: {e}")
            # Instead of returning None, raise the exception
            raise

    def extract_label(self, img_path):
        original_label = None
    
        if "data/WIDER/Image/train" in img_path:
            label_str = img_path.split("data/WIDER/Image/train/")[1].split("/")[0]
            original_label = int(label_str.split("--")[0])
        elif "data/WIDER/Image/test" in img_path:
            label_str = img_path.split("data/WIDER/Image/test/")[1].split("/")[0]
            original_label = int(label_str.split("--")[0])
        elif "data/WIDER/Image/val" in img_path:  # Handle validation images
            label_str = img_path.split("data/WIDER/Image/val/")[1].split("/")[0]
            original_label = int(label_str.split("--")[0])
    
        if original_label is not None:
            remapped_label = self.label_mapping[original_label]
            return remapped_label
        else:
            raise ValueError(f"Label could not be extracted from path: {img_path}")

###

def load_wider(train_file_path, test_file_path, class_labels, batch_size, num_workers):
    train_file = [train_file_path]
    test_file = [test_file_path]

    train_dataset = DataSet(train_file, class_labels, augs = ['RandAugment'], img_size = 226, dataset = 'wider')
    test_dataset = DataSet(test_file, class_labels, augs = [], img_size = 226, dataset = 'wider')


    def custom_collate(batch):
        # Filter out any None items in the batch
        batch = [item for item in batch if item is not None]
        # If after filtering the batch is empty, handle this case by either returning an empty tensor or raising an exception
        if len(batch) == 0:
            # Option 1: Return a placeholder tensor (adapt the shape to match your data)
            # return torch.tensor([]), torch.tensor([])
            # Option 2: Raise an exception
            raise ValueError("Batch is empty after filtering out None items.")
        return torch.utils.data.dataloader.default_collate(batch)
    
    

    trainloader = DataLoader(train_dataset, 
                          batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, collate_fn=custom_collate)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=custom_collate)

    return trainloader, testloader


