import argparse
import time


def arg_parse():
     parser = argparse.ArgumentParser(description='Canvas ImageNet trainer/searcher')

     # Models.
     parser.add_argument('--model', type=str, metavar='NAME', required=True,
                         help='Name of model to train (default: "resnet18"')
     parser.add_argument('--num-classes', type=int, metavar='N',
                         help='Number of label classes (model default if none)')
     parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                         help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc, model default if none)')
     parser.add_argument('--img-size', type=int, default=None, metavar='N',
                         help='Image patch size (model default if none)')
     parser.add_argument('--input-size', default=(3, 224, 224), nargs=3, type=int, metavar='N N N',
                         help='Input all image dimensions (d h w, e.g. --input-size 3 224 224, '
                              'model default if none)')
     parser.add_argument('--crop-pct', default=None, type=float,
                         metavar='N', help='Input image center crop percent (for validation only)')
     parser.add_argument('--mean', type=float, nargs='+', default=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), metavar='MEAN',
                         help='Override mean pixel value of dataset')
     parser.add_argument('--std', type=float, nargs='+', default=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404), metavar='STD',
                         help='Override std deviation of of dataset')
     parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                         help='Image resize interpolation type (overrides model)')
     parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                         help='Dropout rate (default: 0.)')
     parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                         help='Drop path rate (default: None)')
     parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                         help='Drop block rate (default: None)')
     parser.add_argument('--need-model_complexity_info', default=False, action='store_true', help='The complexity info will be given if used')

     # Dataset.
     parser.add_argument('--seed', type=int, default=42, metavar='S',
                         help='Random seed (default: 42)')
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
     parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                         help='Use the multi-epochs-loader to save time at the beginning of every epoch')

     # Dataset augmentation.
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
                         help='Mixup alpha, mixup enabled if > 0. (default: 0.8)')
     parser.add_argument('--cutmix', type=float, default=1.0,
                         help='Cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
     parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                         help='Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
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
     parser.add_argument('--tta', type=int, default=0, metavar='N',
                         help='Test/inference time augmentation (oversampling) factor')

     # Loss functions.
     parser.add_argument('--jsd-loss', action='store_true', default=False,
                         help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
     parser.add_argument('--bce-loss', action='store_true', default=False,
                         help='Enable BCE loss w/ Mixup/CutMix use.')
     parser.add_argument('--bce-target-thresh', type=float, default=None,
                         help='Threshold for binarizing softened BCE targets (default: None, disabled)')

     # BatchNorm settings.
     parser.add_argument('--dist-bn', type=str, default='reduce',
                         help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

     # Optimizer for weight parameters.
     parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                         help='Learning rate (default: 1e-3)')
     parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                         help='Optimizer (default: "adamw"')
     parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                         help='Optimizer Epsilon (default: None, use opt default)')
     parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                         help='Optimizer Betas (default: None, use opt default)')
     parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                         help='Optimizer momentum (default: 0.9)')
     parser.add_argument('--weight-decay', type=float, default=0.05,
                         help='Weight decay (default: 0.05)')
     parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                         help='Clip gradient norm (default: none, no clipping)')
     parser.add_argument('--clip-mode', type=str, default='agc',
                         help='Gradient clipping mode, one of ("norm", "value", "agc")')

     # Scheduler parameters.
     parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                         help='LR scheduler (default: "step"')
     parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='PCT, PCT',
                         help='Learning rate noise on/off epoch percentages')
     parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='Learning rate noise limit percent (default: 0.67)')
     parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='Learning rate noise std-dev (default: 1.0)')
     parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                         help='Learning rate cycle len multiplier (default: 1.0)')
     parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                         help='Amount to decay each learning rate cycle (default: 0.5)')
     parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                         help='Learning rate cycle limit, cycles enabled if > 1')
     parser.add_argument('--lr-k-decay', type=float, default=1.0,
                         help='Learning rate k-decay for cosine/poly (default: 1.0)')
     parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                         help='Warmup learning rate (default: 1e-6)')
     parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                         help='Lower lr bound for cyclic schedulers that hit 0 (1e-5)')
     parser.add_argument('--epochs', type=int, default=300, metavar='N',
                         help='Number of epochs to train (default: 300)')
     parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                         help='Epoch interval to decay LR')
     parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                         help='Epochs to warmup LR, if scheduler supports')
     parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                         help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
     parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                         help='Patience epochs for Plateau LR scheduler (default: 10')
     parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                         help='LR decay rate (default: 0.1)')
          
     # Misc.
     parser.add_argument('--needs-profiler', default=False, action='store_true', help='Enable torch profiler')
     parser.add_argument('--forbid-eval-nan', action='store_true', help='Whether to forbid NaN during evaluation')
     parser.add_argument('--output', default='', type=str, metavar='PATH',
                         help='Path to output folder (default: none, current dir, training only)')
     parser.add_argument('--resume', metavar='PATH', type=str, default='',
                         help='Path to the checkpoint for resuming (only for training)')
     parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                         help='Number of checkpoints to keep (default: 10)')
     parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                         help='Best metric (default: "top1"')
     parser.add_argument('--log-interval', default=100, type=int, metavar='INTERVAL',
                         help='Logging interval')

     # Distributed.
     parser.add_argument("--local_rank", default=0, type=int)
     parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                         help='Force broadcast buffers for native DDP to off')

     # Canvas preferences.
     parser.add_argument('--canvas-load-checkpoint', default='', type=str,
                         help='Path to checkpoint file (after replacing the kernel)')
     parser.add_argument('--canvas-rounds', default=0, type=int,
                         help='Search rounds for Canvas (only for search)')
     parser.add_argument('--canvas-seed', default='pure', type=str,
                         help='Canvas seed settings (one of "global" and "pure"), '
                              '"global" means same with training, "pure" means purely random '
                              '(only for search)')
     parser.add_argument('--canvas-first-epoch-pruning-milestone', default='', type=str,
                         help='First epoch milestone pruning')
     parser.add_argument('--canvas-epoch-pruning-milestone', default='', type=str,
                         help='Epoch accuracy milestone pruning')
     parser.add_argument('--canvas-log-dir', default='', type=str,
                         help='Canvas logging directory')
     parser.add_argument('--canvas-tensorboard-log-dir', default=None, type=str,
                         help='Canvas logging directory using tensorboard')
     parser.add_argument('--canvas-oss-bucket', default='', type=str,
                         help='Log into OSS buckets')
     parser.add_argument('--canvas-bmm-pct', default=0.1, type=float,
                         help='Possibility to forcibly contain BMM (attention-like, only for search)')
     parser.add_argument('--canvas-proxy-root', default='', metavar='DIR', type=str,
                         help='Path to proxy dataset (only for search)')
     parser.add_argument('--canvas-proxy-threshold', default=70.0, type=float,
                         help='Proxy dataset threshold for real training (only for search)')
     parser.add_argument('--canvas-kernels', type=str, nargs='+', default=[],
                         help='Path to the replaced kernel (only for training)')
     parser.add_argument('--canvas-min-macs', default=0, type=float,
                         help='Minimum MACs for searched kernels (in G-unit, only for search)')
     parser.add_argument('--canvas-max-macs', default=0, type=float,
                         help='Maximum MACs for searched kernels (in G-unit, only for search)')
     parser.add_argument('--canvas-min-params', default=0, type=float,
                         help='Minimum params for searched kernels (in M-unit, only for search)')
     parser.add_argument('--canvas-max-params', default=0, type=float,
                         help='Maximum params for searched kernels (in M-unit, only for search)')
     parser.add_argument('--canvas-min-receptive-size', default=1, type=int,
                         help='Minimum receptive size (only for search)')
     parser.add_argument('--canvas-min-proxy-kernel-scale', default=0.2, type=float,
                         help='Minimum kernel scale (geometric mean, only for search)')
     parser.add_argument('--canvas-sampling-workers', default=10, type=int,
                         help='Workers to use for sampling (only for search)')
     parser.add_argument('--canvas-proxy-kernel-scale-limit', default=0.3, type=float,
                         help='Minimum/maximum kernel scale (geometric mean, only for search)')
     parser.add_argument('--canvas-selector-address', default='http://43.138.119.173:8000', type=str,
                         help='Selector server address')
     parser.add_argument('--canvas-selector-max-params', default=6, help='Maximum model size')
     parser.add_argument('--canvas-selector-dir', default='', help='Selector working directory')
     parser.add_argument('--canvas-selector-save-dir', default='', help='Selector saving directory')
     parser.add_argument('--canvas-number-of-kernels', default=4, type = int, help='The number of kernels inside the replaced module')
     parser.add_argument('--compression-rate', default=1.0, help='The compression rate after replaced with replaced module')

     # Proxyless mode
     parser.add_argument('--proxyless', action='store_true', default=False)
     parser.add_argument('--needs-valid', action='store_true', default=False)
     parser.add_argument('--alpha-lr', type=float, default=0.001)
     parser.add_argument('--alpha-weight-decay', type=float, default=0.0)
     parser.add_argument('--num-iters-update-alphas', type=int, default=5)
     parser.add_argument('--alpha_update_steps', type=int, default=1)
     parser.add_argument('--grad_data_batch', type=int, default=None)

     # Parse program arguments, add timestamp information, and checks.
     args = parser.parse_args()
     setattr(args, 'timestamp', time.time_ns())
     assert args.canvas_min_macs <= args.canvas_max_macs, 'Minimum FLOPs should be lower than maximum'
     assert args.canvas_min_params <= args.canvas_max_params, 'Minimum params should be lower than maximum'

     return args
