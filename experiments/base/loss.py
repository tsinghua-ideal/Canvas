from torch import nn

from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


def get_loss_funcs(args):
    def get_train_loss():
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            return BinaryCrossEntropy(target_threshold=args.bce_target_thresh) \
                if args.bce_loss \
                else SoftTargetCrossEntropy()
        elif args.smoothing:
            return BinaryCrossEntropy(target_threshold=args.bce_target_thresh) \
                if args.bce_loss \
                else LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        return nn.CrossEntropyLoss()

    return get_train_loss().cuda(), nn.CrossEntropyLoss().cuda()
