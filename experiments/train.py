from base import dataset, device, loss, models, optim, parser, sche, trainer


if __name__ == '__main__':
    # Check available devices.
    device.initialize()

    # Get arguments.
    args = parser.arg_parse()

    # Training utils.
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)
    loss_funcs = loss.get_loss_funcs(args)
    optimizer = optim.get_optimizer(args, model)
    schedule = sche.get_schedule(args, optimizer)

    # Start to train!
    trainer.train(args, model=model,
                  train_loader=train_loader, eval_loader=eval_loader,
                  loss_funcs=loss_funcs, optimizer=optimizer,
                  schedule=schedule)
