import itertools
import canvas
import random
import torch

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

    # Set up Canvas randomness seed.
    logger.info(f'Configuring Canvas ...')
    canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)

    # Initialization of search.
    best_score, best_clone = 0, canvas.get_state_dict(model, remove_placeholders=True)
    if args.load_checkpoint:
        best_score = 100  # Always re-train from checkpoint if specified.

    # Search.
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    for i in round_range:
        # Sample a new kernel.
        logger.info('Sampling a new kernel ...')
        try:
            kernel_pack = canvas.sample(model, force_bmm_possibility=args.canvas_bmm_pct)
            canvas.replace(model, kernel_pack.module, args.device)
        except RuntimeError as ex:
            # Out of memory or timeout.
            logger.warning(f'Exception: {ex}')
            continue
        logger.info('Sampled kernel hash: {}'.format(hash(kernel_pack)))

        # Train.
        proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
        try:
            if proxy_train_loader and proxy_eval_loader:
                logger.info('Training on proxy dataset ...')
                _, proxy_eval_metrics = \
                    trainer.train(args, model=model, train_loader=proxy_train_loader, eval_loader=proxy_eval_loader)
                proxy_score = max([item['top1'] for item in proxy_eval_metrics])
                canvas.restore_from_state_dict(model, best_clone)
                logger.info(f'Proxy dataset score: {proxy_score}')
                if proxy_score < args.canvas_proxy_threshold:
                    logger.info(f'Under proxy threshold {args.canvas_proxy_threshold}, skip main dataset training')
                    continue
            logger.info('Training on main dataset ...')
            train_metrics, eval_metrics = \
                trainer.train(args, model=model, train_loader=train_loader, eval_loader=eval_loader)
            score = max([item['top1'] for item in eval_metrics])
            logger.info(f'Solution score: {score}')
            if score > best_score:
                best_score, best_clone = score, canvas.get_state_dict(model, remove_placeholders=True)
                logger.info(f'Best score has been updated to {best_score}')
            else:
                logger.warning('Restoring to best model parameters')
                canvas.restore_from_state_dict(model, best_clone)
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            if 'NaN' in exception_info:
                logger.warning('Restoring to best model parameters')
                canvas.restore_from_state_dict(model, best_clone)

        # Save into logging directory.
        extra = {'proxy_score': proxy_score}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, kernel_pack, train_metrics, eval_metrics, extra)
