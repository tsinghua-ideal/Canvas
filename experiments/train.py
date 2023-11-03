import torch
import os, json
from base import dataset, device, models, parser, trainer, log, proxyless_trainer


if __name__ == '__main__':
    # Get arguments
    logger = log.get_logger()
    args = parser.arg_parse()
    
    # Check available devices, set seed, and setdistributed
    device.initialize(args)
    
    # Training utils
    model = models.get_model(args)
    if args.needs_valid:
        train_loader, valid_loader, eval_loader = dataset.get_loaders(args)
    else:
        train_loader, eval_loader = dataset.get_loaders(args)
        
    # Start to train
    if args.needs_valid:
        metrics = proxyless_trainer.train(args, model=model, train_loader=train_loader, valid_loader=valid_loader, eval_loader=eval_loader)
    else:
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