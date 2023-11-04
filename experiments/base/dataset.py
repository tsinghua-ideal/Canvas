import logging
import sys
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import transforms


def get_next_valid_sample(loader):
    while True:
        for sample in loader:
            yield sample


def make_random_square_masks(inputs, mask_size):
    if mask_size == 0:
        return None
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even)
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even)

    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

    final_mask = to_mask_y * to_mask_x
    return final_mask


@torch.no_grad()
def batch_cutout(inputs, patch_size):
    cutout_batch_mask = make_random_square_masks(inputs, patch_size)
    inputs = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)
    return inputs


@torch.no_grad()
def batch_crop(inputs, crop_size):
    crop_mask_batch = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
    return cropped_batch


@torch.no_grad()
def batch_flip_lr(batch_images, flip_chance=.5):
    return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


@torch.no_grad()
def get_batches(data_dict, key, batch_size, crop_size):
    num_epoch_examples = len(data_dict['images'])
    shuffled = torch.randperm(num_epoch_examples, device='cuda')

    if key == 'train' or key == 'valid':
        images = batch_crop(data_dict['images'], crop_size)
        images = batch_flip_lr(images)
        images = batch_cutout(images, patch_size=3)
        train_transformer = transforms.Compose([
            # transforms.RandomResizedCrop(size=(32, 32), scale=(0.95,1.0), antialias=True),
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            # transforms.RandomRotation(10),       # Randomly rotate the image by up to 10 degrees
            # transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust brightness, contrast, saturation, and hue
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Apply a random affine transformation
            transforms.RandomPerspective(),     # Apply a random perspective transformation
        ])
        images = train_transformer(images)

    else:
        images = data_dict['images']
    labels = data_dict['labels']

    for idx in range(num_epoch_examples // batch_size):
        if not (idx + 1) * batch_size > num_epoch_examples:
            x, y = images.index_select(0, shuffled[idx * batch_size:(idx + 1) * batch_size]), labels.index_select(0, shuffled[idx * batch_size:(idx + 1) * batch_size])
            x = F.interpolate(x, size=(224, 224), mode='bilinear')
            yield x, y


class FuncDataloader():
    def __init__(self, func, len):
        self.func = func
        self.len = len
        
    def __iter__(self):
        return self.func()
    
    def __len__(self):
        return self.len


def get_loaders(args):
    # Get tensors
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = str(args.dataset).upper()
    assert hasattr(sys.modules[__name__], dataset_name), f'Could not find dataset {args.dataset}'
    func = getattr(sys.modules[__name__], dataset_name)
    train_data = func(root=args.root, download=True, train=True, transform=transform)
    eval_data = func(root=args.root, download=False, train=False, transform=transform)   
    if args.needs_valid:
        train_size = int(args.proportion_of_training_set * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = random_split(train_data, [train_size, valid_size])
        logging.info(f'train_dataset: {len(train_data)}, valid_dataset: {len(valid_data)}')

    # Get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=False) if args.needs_valid else None
    eval_dataloader = DataLoader(eval_data, batch_size=len(eval_data), shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    
    # Compose to datasets
    train_dataset, valid_dataset, eval_dataset = {}, {}, {}
    train_dataset['images'], train_dataset['labels'] = [item.cuda(non_blocking=True) for item in next(iter(train_dataloader))]
    valid_dataset['images'], valid_dataset['labels'] = [item.cuda(non_blocking=True) for item in next(iter(valid_dataloader))] if args.needs_valid else (None, None)
    eval_dataset['images'],  eval_dataset['labels']  = [item.cuda(non_blocking=True) for item in next(iter(eval_dataloader)) ]
    
    # Normalize images
    train_std, train_mean = torch.std_mean(train_dataset['images'], dim=(0, 2, 3))
    valid_std, valid_mean = torch.std_mean(valid_dataset['images'], dim=(0, 2, 3)) if args.needs_valid else (None, None)
    eval_std, eval_mean = torch.std_mean(eval_dataset['images'], dim=(0, 2, 3))
    logging.info(f'train_std: {train_std}, train_mean: {train_mean}')
    if args.needs_valid:
        logging.info(f'valid_std: {valid_std}, valid_mean: {valid_mean}')
    logging.info(f'eval_std: {eval_std}, eval_mean: {eval_mean}')
    
    def batch_normalize_images(input_images, mean, std):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    
    train_batch_normalize_images = partial(batch_normalize_images, mean=train_mean, std=train_std)
    valid_batch_normalize_images = partial(batch_normalize_images, mean=valid_mean, std=valid_std) if args.needs_valid else None
    eval_batch_normalize_images = partial(batch_normalize_images, mean=eval_mean, std=eval_std)
    
    train_dataset['images'] = train_batch_normalize_images(train_dataset['images'])
    valid_dataset['images'] = valid_batch_normalize_images(valid_dataset['images']) if args.needs_valid else None
    eval_dataset['images']  = eval_batch_normalize_images(eval_dataset['images'])


    # Padding
    assert train_dataset['images'].shape[-1] == train_dataset['images'].shape[-2], 'Images must be square'
    train_crop_size = int(train_dataset['images'].shape[-1] * args.scale)
    train_dataset['images'] = F.pad(train_dataset['images'], (2, ) * 4, 'reflect')
    if args.needs_valid:
        assert valid_dataset['images'].shape[-1] == valid_dataset['images'].shape[-2], 'Images must be square'
        valid_crop_size = int(valid_dataset['images'].shape[-1] * args.scale)
        valid_dataset['images'] = F.pad(valid_dataset['images'], (2, ) * 4, 'reflect')
    # TODO: Add random crop to eval_dataset
    eval_crop_size = eval_dataset['images'].shape[-1]
    
    train_loader = FuncDataloader(partial(get_batches, data_dict=train_dataset, key='train', batch_size=args.batch_size, crop_size=train_crop_size), len=len(train_dataset['images']) // args.batch_size)
    valid_loader = FuncDataloader(partial(get_batches, data_dict=valid_dataset, key='valid', batch_size=args.batch_size, crop_size=valid_crop_size), len=len(valid_dataset['images']) // args.batch_size) if args.needs_valid else None
    eval_loader = FuncDataloader(partial(get_batches, data_dict=eval_dataset, key='eval', batch_size=args.batch_size, crop_size=eval_crop_size), len(eval_dataset['images']) // args.batch_size)
    return (train_loader, valid_loader, eval_loader) if args.needs_valid else (train_loader, eval_loader)