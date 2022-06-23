import itertools
import canvas
import random
import torch

from base import dataset, device, log, models, parser, trainer


if __name__ == '__main__':
    # Get arguments.
    args = parser.arg_parse()

    # Check available devices and set distributed.
    device.initialize(args)
    assert not args.distributed, 'Search mode does not support distributed training'

    # Training utils.
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)

    # Set up Canvas randomness seed.
    canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)

    # Initialization.
    example_input = torch.zeros((1, ) + args.input_size).to(args.device)
    canvas.get_placeholders(model, example_input)
    best_score, best_clone = 0, canvas.get_state_dict(model, remove_placeholders=True)

    # Search.
    logger = log.get_logger()
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    for i in round_range:
        # Sample a new kernel.
        logger.info('Sampling a new kernel ...')
        try:
            kernel_pack = canvas.sample(model, example_input)
            canvas.replace(model, kernel_pack.module, args.device)
        except RuntimeError as ex:
            # Out of memory or timeout.
            logger.warning(f'Exception: {ex}')
            continue
        logger.info('Sampled kernel hash: {}'.format(kernel_pack.hash))

        # Train.
        train_metrics, eval_metrics, exception_info = None, None, None
        try:
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
        log.save(args, kernel_pack, train_metrics, eval_metrics, exception_info)
