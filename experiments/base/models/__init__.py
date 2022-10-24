import canvas
import torch
import ptflops
import timm
from timm import data

from .canvas_van import canvas_van_tiny
from ..log import get_logger


def get_model(args, search_mode: bool = False):
    logger = get_logger()
    model = timm.create_model(args.model,
                              num_classes=args.num_classes,
                              drop_rate=args.drop,
                              drop_path_rate=args.drop_path,
                              drop_block_rate=args.drop_block)
    model.to(args.device)

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

    # Initialize placeholders.
    example_input = torch.zeros((1, ) + args.input_size).to(args.device)
    canvas.get_placeholders(model, example_input)

    # Replace kernel.
    if not search_mode and args.canvas_kernel:
        if args.local_rank == 0:
            logger.info(f'Replacing kernel from {args.canvas_kernel}')
        pack = canvas.KernelPack.load(args.canvas_kernel)
        model = canvas.replace(model, pack.module, args.device)

    # Count FLOPs and params.
    if args.local_rank == 0:
        macs, params = ptflops.get_model_complexity_info(model, args.input_size, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
        logger.info(f'MACs: {macs}, params: {params}')

    return model
