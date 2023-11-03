from torch import nn


def get_loss_funcs(args):
    def get_train_loss():
        return nn.CrossEntropyLoss()

    if args.needs_valid:
        return get_train_loss().cuda(), get_train_loss().cuda(), nn.CrossEntropyLoss().cuda()
    else:
        return get_train_loss().cuda(), nn.CrossEntropyLoss().cuda()
