from base import dataset, device, loss, models, optim, parser, sched


if __name__ == '__main__':
    # Check available devices.
    device.check_available()

    # Get arguments.
    args = parser.arg_parse()

    # Training utils.
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)
    loss_func = loss.get_loss_func(args)
    optimizer = optim.get_optimizer(args, model)
    scheduler = sched.get_scheduler(args, optimizer)
