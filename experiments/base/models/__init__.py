import timm

from .van import *


def get_instance(args):
    return timm.create_model(args.model,
                             num_classes=args.num_classes,
                             drop_rate=args.drop,
                             drop_path_rate=args.drop_path,
                             drop_block_rate=args.drop_block)
