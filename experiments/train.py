from base import dataset, loss, models, optim, parser


if __name__ == '__main__':
    args = parser.arg_parse()
    model = models.get_model(args)
    train_loader, eval_loader = dataset.get_loaders(args)
    loss_func = loss.get_loss_func(args)
    optimizer = optim.get_optimizer(args, model)
    print(locals())
