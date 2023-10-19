import torch
import os, json
from base import dataset, device, models, parser, trainer, log


if __name__ == '__main__':
    # Get arguments
    logger = log.get_logger()
    args = parser.arg_parse()
    # Check available devices and set distributed
    device.initialize(args)
    
    # Training utils
    model = models.get_model(args)
    train_loader,  eval_loader = dataset.get_loaders(args)
    
    # Start to train
    metrics = trainer.train(args, model=model, train_loader=train_loader, eval_loader=eval_loader)
    if os.path.exists(args.canvas_log_dir):
            assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
    dir_name = f'Canvas_warmup_epochs_{args.warmup_epochs}_epochs_{args.epochs}_lr_{args.lr}'
    path = os.path.join(args.canvas_log_dir, dir_name)
    if os.path.exists(path):
        logger.info('Overwriting results ...')
    os.makedirs(path, exist_ok=True)

    # Save args and metrics.
    with open(os.path.join(path, 'metrics.json'), 'w') as file:
        json.dump({'args': vars(args), 'metrics': metrics
                },
                fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)