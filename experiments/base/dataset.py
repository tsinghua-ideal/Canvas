import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


supported_dataset = {
    'CIFAR10': [10, '/scorpio/home/zhaocg/Datasets/CIFAR'],
    'CIFAR100': [100, '/scorpio/home/zhaocg/Datasets/CIFAR'],
    'Imagewoof': [10, '/scorpio/home/zhaocg/Datasets/imagewoof2/'],
    'Imagenette': [10, '/scorpio/home/zhaocg/Datasets/imagenette2/']
}


def get_train_dataset(name: str, path: str = None,
                      batch_size: int = 512, shuffle: bool = True):
    assert name in supported_dataset, f'Dataset {name} not supported'
    path = supported_dataset[name][1] if not path else path
    dataset = None

    if name in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(224),
        ])
        dataset = None
        if name == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=path, train=True, transform=transform)
        if name == 'CIFAR100':
            dataset = torchvision.datasets.CIFAR100(root=path, train=True, transform=transform)
    elif name in ['Imagewoof', 'Imagenette']:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = torchvision.datasets.ImageFolder(root=path + '/train', transform=transform)

    # Return dataset
    assert dataset is not None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    return supported_dataset[name][0], dataloader


def get_test_dataset(name: str, path: str = None, batch_size: int = 256):
    assert name in supported_dataset, f'Dataset {name} not supported'
    path = supported_dataset[name][1] if not path else path
    dataset = None

    if name in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(224),
        ])
        dataset = None
        if name == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=transform)
        if name == 'CIFAR100':
            dataset = torchvision.datasets.CIFAR100(root=path, train=False, transform=transform)
    elif name in ['Imagewoof', 'Imagenette']:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = torchvision.datasets.ImageFolder(root=path + '/val', transform=transform)

    # Return dataset
    assert dataset is not None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return supported_dataset[name][0], dataloader


def get_single_sample(dataloader, keep_dim: bool = False):
    images, labels = next(iter(dataloader))
    images, labels = images[0], labels[0]
    if keep_dim:
        images, labels = torch.unsqueeze(images, 0), torch.unsqueeze(labels, 0)
    return images, labels


def get_batch(dataloader):
    images, labels = next(iter(dataloader))
    return images, labels
