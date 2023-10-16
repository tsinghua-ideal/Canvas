import random
import torch
from torch.utils.data import random_split
from timm.data import create_dataset, FastCollateMixup, create_loader

from .log import get_logger

def get_next_valid_sample(loader):
    while True:
        for sample in loader:
            
            yield sample

def get_loaders(args, proxy: bool = False):
    # Whether proxy.
    root = args.canvas_proxy_root if proxy else args.root
    if root == '':
        return None, None
    logger = get_logger()
    # Get dataset.
    if args.local_rank == 0:
        get_logger().info(f'Preparing data loaders in {root} (proxy={proxy})') 

    dataset_train = create_dataset(name='torch/cifar10', root=root, split=args.train_split, download=True)
    dataset_eval = create_dataset(name='torch/cifar10', root=root, split=args.val_split, download=True)
    
    if args.needs_valid:
        torch.manual_seed(42)
        train_size = int(0.85 * len(dataset_train))
        valid_size = len(dataset_train) - train_size
        dataset_train, dataset_valid = random_split(dataset_train, [train_size, valid_size])
        logger.info(f'train_dataset: {len(dataset_train)}, valid_dataset: {len(dataset_valid)}')

    # Setup mixup / cutmix.
    collate_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        collate_fn = FastCollateMixup(**mixup_args)
        
    # Create data loaders w/ augmentation pipeline.
    train_loader = create_loader(
        dataset_train.dataset if args.needs_valid else dataset_train,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=args.no_aug,
        re_prob=args.re_prob,
        re_mode=args.re_mode,
        re_count=args.re_count,
        re_split=args.re_split,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        # num_aug_repeats=aug_repeats,
        # num_aug_splits=0,
        interpolation=args.train_interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        device=torch.device(args.device)
    )
    
    if args.needs_valid:
        valid_loader = create_loader(
            dataset_valid.dataset,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=args.no_aug,
            re_prob=args.re_prob,
            re_mode=args.re_mode,
            re_count=args.re_count,
            re_split=args.re_split,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            # num_aug_repeats=aug_repeats,
            # num_aug_splits=0,
            interpolation=args.train_interpolation,
            mean=args.mean,
            std=args.std,
            num_workers=args.num_workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            device=torch.device(args.device)
        )

    eval_loader = create_loader(
        dataset_eval,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=args.crop_pct,
        pin_memory=args.pin_memory,
        device=torch.device(args.device)
    )
    
    return (train_loader, valid_loader, eval_loader) if args.needs_valid else (train_loader, eval_loader)

