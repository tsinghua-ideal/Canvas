import torch

from base import dataset, device, models, parser, trainer, log, proxyless_trainer


if __name__ == '__main__':
    # Get arguments
    logger = log.get_logger()
    args = parser.arg_parse()

    # Check available devices and set distributed
    device.initialize(args)

    # Training utils
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)

    # Load checkpoint
    if args.canvas_load_checkpoint:
        logger.info(f'Loading checkpoint from {args.canvas_load_checkpoint}')
        checkpoint = torch.load(args.canvas_load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Start to train
    (proxyless_trainer if args.proxyless else trainer) \
        .train(args, model=model, train_loader=train_loader, eval_loader=eval_loader)
