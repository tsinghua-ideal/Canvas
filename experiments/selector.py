import gc
import itertools
import torch
import ptflops

import canvas

from copy import deepcopy
from base import dataset, device, log, models, parser, trainer


if __name__ == '__main__':
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()
    logger.info(f'Program arguments: {args}')

    # Check available devices and set distributed.
    logger.info(f'Initializing devices ...')
    device.initialize(args)
    assert not args.distributed, 'Selector mode does not support distributed training'

    # Training utils.
    logger.info(f'Configuring model {args.model} ...')
    model = models.get_model(args, search_mode=True)
    train_loader, eval_loader = dataset.get_loaders(args)

    # Load checkpoint.
    if args.canvas_load_checkpoint:
        logger.info(f'Loading checkpoint from {args.canvas_load_checkpoint}')
        checkpoint = torch.load(args.canvas_load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Initialization of search.
    cpu_clone = deepcopy(model).cpu()

    def restore_model_params_and_replace(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        if pack is not None:
            canvas.replace(model, pack.module, args.device)

    # Search.
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    for i in round_range:
        # Sample a new kernel.
        logger.info('Requesting a new kernel ...')
        try:
            # TODO: web API.
            kernel_pack = None

            restore_model_params_and_replace(kernel_pack)
            g_macs, m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                 as_strings=False, print_per_layer_stat=False)
            g_macs, m_params = g_macs / 1e9, m_params / 1e6
        except RuntimeError as ex:
            # Early exception: out of memory or timeout.
            logger.info(f'Exception: {ex}')
            log.save(args, None, None, None, {'exception': f'{ex}'})
            # TODO: notify web API (failure).
            continue
        logger.info(f'Sampled kernel hash: {hash(kernel_pack)}')
        logger.info(f'MACs: {g_macs} G, params: {m_params} M')

        # Train.
        train_metrics, eval_metrics, exception_info = None, None, None
        try:
            logger.info('Training on main dataset ...')
            train_metrics, eval_metrics = \
                trainer.train(args, model=model,
                              train_loader=train_loader, eval_loader=eval_loader,
                              search_mode=True)
            score = max([item['top1'] for item in eval_metrics])
            logger.info(f'Solution score: {score}')
            # TODO: notify web API (success).
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            if 'NaN' in exception_info:
                logger.warning('Restoring to best model parameters')
            # TODO: notify web API (failure).

        # Save into logging directory.
        extra = {'g_macs': g_macs, 'm_params': m_params}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, kernel_pack, train_metrics, eval_metrics, extra)
