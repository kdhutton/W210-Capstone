
### How to load data


```
from data.data_loader import load_cifar10, load_cifar100, load_imagenet

train_loader, test_loader = load_cifar10()
train_loader, test_loader  = load_cifar100()
train_loader, test_loader  = load_imagenet('/path_to_imagenet_train', '/path_to_imagenet_test')
```

_Note: FACET dataloader is working in progres_
