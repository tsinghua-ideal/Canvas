import json
import os
import torch
import torch.distributed as dist

from base import dataset, device, models, parser, trainer, log

if __name__ == '__main__':
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()

    # Check available devices and set distributed.
    device.initialize(args)

    def broadcast(obj):
        object_list = [obj]
        dist.broadcast_object_list(object_list)
        return object_list[0]

    # Get all kernels.
    kernel_paths = []
    if args.rank == 0:
        print(f'Reading files in path {args.canvas_selector_dir} ...')
        for name in os.listdir(args.canvas_selector_dir):
            kernel_path = os.path.join(args.canvas_selector_dir, name)
            if os.path.isdir(kernel_path):
                if len(name.split('_')) != 4:
                    continue
                _, s, k, h = name.split('_')
                files = [name for name in os.listdir(kernel_path)]
                if set(files) != {f'{k}_{h}.dot', f'{k}_{h}.json', f'{k}_{h}.py'}:
                    continue
                with open(os.path.join(kernel_path, f'{k}_{h}.json')) as f:
                    j = json.load(f)
                    if j['extra']['m_params'] <= args.canvas_selector_max_params:
                        kernel_paths.append(kernel_path)
        kernel_paths = sorted(kernel_paths, reverse=True)
        print(f'{len(kernel_paths)} valid kernels (param threshold: {args.canvas_selector_max_params}) collected.')
    else:
        kernel_paths = []
    kernel_paths = broadcast(kernel_paths)

    # Evaluate all the kernels.
    best_score = 0
    for kernel_path in kernel_paths:
        # Train
        if args.rank == 0:
            print(f'Evaluating kernel: {kernel_path}')
        args.canvas_kernel = kernel_path
        model = models.get_model(args)
        train_loader, eval_loader = dataset.get_loaders(args)
        # TODO: pruner with epoch accuracy.
        train_metrics, eval_metrics = \
            trainer.train(args, model=model,
                          train_loader=train_loader, eval_loader=eval_loader,
                          search_mode=True)
        score = max([item['top1'] for item in eval_metrics])
        if args.local_rank == 0:
            logger.info(f'Solution score: {score}')
        if score > best_score:
            best_score = score
        if args.local_rank == 0:
            logger.info(f'Best score: {best_score}')

        # Barrier for the next and move results into another directory.
        if args.local_rank == 0 and args.canvas_selector_save_dir:
            logger.info(f'Moving {kernel_path} into {args.canvas_selector_save_dir}')
            # TODO: move directory.
        dist.barrier()

    # Exit.
    if args.rank == 0:
        logger.info('Finish all the kernels, exit')