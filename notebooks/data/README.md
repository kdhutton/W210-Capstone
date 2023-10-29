
### How to load data

_Note: FACET dataloader is working in progres_


```
from data.data_loader import load_cifar10, load_cifar100, load_imagenet

train_loader, test_loader = load_cifar10()
train_loader, test_loader  = load_cifar100()
train_loader, test_loader  = load_imagenet('/path_to_imagenet_train', '/path_to_imagenet_test')
train_loader, test_loader = load_coco(data_dir)
```

For the coco data direction, please follow the instruction below to download data.

```
pip3 install awscli
aws configure
mkdir /home/ubuntu/W210-Capstone/notebooks/data_coco
aws s3 sync s3://210bucket/coco/ /home/ubuntu/W210-Capstone/notebooks/data_coco
```
__Follow the [instruction](https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to download Imagenet Dataset__


