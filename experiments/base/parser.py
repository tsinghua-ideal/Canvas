import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Canvas ImageNet trainer/searcher')

    # Models.
    parser.add_argument('--model', type=str, metavar='NAME', default='resnet18',
                        help='Name of model to train (default: "resnet18"')
    parser.add_argument('--num-classes', type=int, metavar='N', required=True,
                        help='Number of label classes (Model default if None)')
    parser.add_argument('--input-size', default=(3, 224, 224), nargs=3, type=int, metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Dataset.
    parser.add_argument('--root', metavar='DIR', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--train-split', metavar='NAME', type=str, default='train',
                        help='Dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', type=str, default='validation',
                        help='Dataset validation split (default: validation)')
    parser.add_argument('--batch-size', metavar='N', type=int, default=128,
                        help='Batch size')
    parser.add_argument('-j', '--num-workers', type=int, default=8, metavar='N',
                        help='How many training processes to use (default: 8)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Dataset augmentation
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--re-prob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--re-mode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--re-count', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--re-split', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')

    # Loss functions.
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')

    # Parse program arguments.
    return parser.parse_args()
