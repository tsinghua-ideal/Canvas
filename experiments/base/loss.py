from torch import nn

from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


def get_loss_func(args):
    def get_loss_func_impl():
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

    return get_loss_func_impl().cuda()
