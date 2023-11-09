import canvas
import torch
import ptflops
import timm
from functools import partial
from timm import data

from .canvas_van import van_b0, compact_van_b0
from .proxyless import ParallelKernels
from ..log import get_logger
from ..darts import replace_module_with_placeholder


def get_model(args):
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
    example_input = torch.zeros((2, ) + args.input_size).to(args.device)
    if len(canvas.get_placeholders(model, example_input)) == 0 and args.needs_replace:
        module_dict = {
            torch.nn.Conv2d: "conv"
        }
        logger.info(f'No placeholders found, replacing modules: {module_dict} with Placeholders')
        replaced, not_replaced = replace_module_with_placeholder(model, module_dict)
        logger.info(f'Replaced {replaced} modules with placeholders, {not_replaced} modules not replaced.')
        canvas.get_placeholders(model, example_input)
    
    # Replace kernel.
    if not args.search_mode and len(args.canvas_kernels) > 0 and args.needs_replace:
        if args.local_rank == 0:
            logger.info(f'Replacing kernel from {args.canvas_kernels}')
        assert len(args.canvas_kernels) == 1 or (len(args.canvas_kernels) > 1 and args.proxyless)
        packs = [canvas.KernelPack.load(kernel) for kernel in args.canvas_kernels]
        cls = packs[0].module if len(packs) == 1 else \
            partial(ParallelKernels, kernel_cls_list=[pack.module for pack in packs])
        model = canvas.replace(model, cls, args.device)    
    
    # Count FLOPs and params.
    if args.local_rank == 0 and not args.proxyless:
        g_macs, m_params = ptflops.get_model_complexity_info(model, args.input_size, as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
        logger.info(f'MACs: {g_macs}, params: {m_params}')

    if args.need_model_complexity_info:
        return model, g_macs, m_params
    else:
        return model
