import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import timm
from timm import data

from .canvas_van import canvas_van_tiny, canvas_van_small, canvas_van_base, canvas_van_large
from .van import van_tiny, van_small, van_base, van_large
from .. import log


def get_model(args):
    model = timm.create_model(args.model,
                              num_classes=args.num_classes,
                              drop_rate=args.drop,
                              drop_path_rate=args.drop_path,
                              drop_block_rate=args.drop_block)
    model.cuda()

    if args.torchscript:
        model = torch.jit.script(model)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    # Resolve data configurations.
    data_config = data.resolve_data_config(vars(args), model=model)

    # Rewrite args.
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    setattr(args, 'train_interpolation', train_interpolation)
    setattr(args, 'interpolation', data_config['interpolation'])
    setattr(args, 'input_size', data_config['input_size'])
    setattr(args, 'mean', data_config['mean'])
    setattr(args, 'std', data_config['std'])
    setattr(args, 'crop_pct', data_config['crop_pct'])

    if args.distributed:
        logger = log.get_logger()
        if args.local_rank == 0:
            logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)

    return model
