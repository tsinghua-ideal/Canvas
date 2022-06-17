from base import dataset, parser


if __name__ == '__main__':
    args = parser.arg_parse()
    train_loader, eval_loader = dataset.get_loaders(args)
    print(train_loader, eval_loader)
