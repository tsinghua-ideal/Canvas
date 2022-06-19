from base import dataset, device, models, parser, trainer


if __name__ == '__main__':
    # Get arguments.
    args = parser.arg_parse()

    # Check available devices and set distributed.
    device.initialize(args)

    # Training utils.
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)

    # Start to train!
    trainer.train(args, model=model, train_loader=train_loader, eval_loader=eval_loader)
