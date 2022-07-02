import gc
import itertools
import torch
import ptflops

import canvas
import random

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
    assert not args.distributed, 'Search mode does not support distributed training'

    # Training utils.
    logger.info(f'Configuring model {args.model} ...')
    model = models.get_model(args, search_mode=True)
    train_loader, eval_loader = dataset.get_loaders(args)
    proxy_train_loader, proxy_eval_loader = dataset.get_loaders(args, proxy=True)

    # Load checkpoint.
    if args.canvas_load_checkpoint:
        logger.info(f'Loading checkpoint from {args.canvas_load_checkpoint}')
        checkpoint = torch.load(args.canvas_load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Set up Canvas randomness seed.
    logger.info(f'Configuring Canvas ...')
    canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)

    # Initialization of search.
    cpu_clone = deepcopy(model).cpu()

    def restore_model_params():
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)

    # Search.
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    for i in round_range:
        restore_model_params()

        # Sample a new kernel.
        logger.info('Sampling a new kernel ...')
        g_macs, m_flops = 0, 0
        try:
            kernel_pack = canvas.sample(model, force_bmm_possibility=args.canvas_bmm_pct)
            canvas.replace(model, kernel_pack.module, args.device)
            g_macs, m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                 as_strings=False, print_per_layer_stat=False)
            g_macs, m_params = g_macs / 1e9, m_params / 1e6
        except RuntimeError as ex:
            # Early exception: out of memory or timeout.
            logger.info(f'Exception: {ex}')
            log.save(args, None, None, None, {'exception': f'{ex}'})
            continue
        logger.info(f'Sampled kernel hash: {hash(kernel_pack)}')
        logger.info(f'MACs: {g_macs} G, params: {m_params} M')
        macs_not_satisfied = (args.canvas_min_macs > 0 or args.canvas_max_macs > 0) and \
                             (g_macs < args.canvas_min_macs or g_macs > args.canvas_max_macs)
        params_not_satisfied = (args.canvas_min_params > 0 or args.canvas_max_params > 0) and \
                               (m_params < args.canvas_min_params or m_params > args.canvas_max_params)
        if macs_not_satisfied or params_not_satisfied:
            logger.info(f'MACs ({args.canvas_min_macs}, {args.canvas_max_macs}) or '
                        f'params ({args.canvas_min_params}, {args.canvas_max_params}) '
                        f'requirements do not satisfy')
            continue

        # Train.
        proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
        try:
            if proxy_train_loader and proxy_eval_loader:
                logger.info('Training on proxy dataset ...')
                _, proxy_eval_metrics = \
                    trainer.train(args, model=model,
                                  train_loader=proxy_train_loader, eval_loader=proxy_eval_loader,
                                  search=True)
                proxy_score = max([item['top1'] for item in proxy_eval_metrics])
                restore_model_params()
                logger.info(f'Proxy dataset score: {proxy_score}')
                if proxy_score < args.canvas_proxy_threshold:
                    logger.info(f'Under proxy threshold {args.canvas_proxy_threshold}, skip main dataset training')
                    continue
            logger.info('Training on main dataset ...')
            train_metrics, eval_metrics = \
                trainer.train(args, model=model,
                              train_loader=train_loader, eval_loader=eval_loader,
                              search=True)
            score = max([item['top1'] for item in eval_metrics])
            logger.info(f'Solution score: {score}')
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            if 'NaN' in exception_info:
                logger.warning('Restoring to best model parameters')

        # Kernel scales after training.
        kernel_scales = None
        if hasattr(model, 'kernel_scales'):
            kernel_scales = model.kernel_scales()
            logger.info(f'Kernel scales after training: {kernel_scales}')

        # Save into logging directory.
        extra = {'proxy_score': proxy_score, 'g_macs': g_macs, 'm_params': m_params, 'kernel_scales': kernel_scales}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, kernel_pack, train_metrics, eval_metrics, extra)
