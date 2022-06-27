import canvas
import json
import logging
import math
import os


logging.basicConfig(level=logging.DEBUG)
_exp_logger = logging.getLogger()
_exp_logger.setLevel(logging.INFO)


def get_logger():
    global _exp_logger
    return _exp_logger


def save(args, kernel_pack: canvas.KernelPack,
         train_metrics, eval_metrics, extra):
    if args.canvas_log_dir:
        # Logging info.
        logger = get_logger()
        logger.info(f'Saving kernel {hash(kernel_pack)} into {args.canvas_log_dir} ...')

        # Make directory (may overwrite).
        if os.path.exists(args.canvas_log_dir):
            assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
        if 'exception' in extra:
            exception_info = extra['exception']
            if 'memory' in exception_info or 'NaN' in exception_info:  # Do not record these types.
                return
            else:
                error_type = 'Error'
            dir_name = f'Canvas_{error_type}_{hash(kernel_pack)}'
        else:
            assert len(eval_metrics) > 0
            max_score = math.floor(max([item['top1'] for item in eval_metrics]) * 100)
            score_str = ('0' * max(0, 5 - len(f'{max_score}'))) + f'{max_score}'
            dir_name = f'Canvas_{score_str}_{hash(kernel_pack)}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save code, graphviz, args, and results.
        kernel_name = f'Kernel_{hash(kernel_pack)}'
        kernel_pack.save_torch_code(os.path.join(path, kernel_name + '.py'))
        kernel_pack.save_graphviz_code(os.path.join(path, kernel_name + '.dot'))
        with open(os.path.join(path, kernel_name + '.json'), 'w') as file:
            json.dump({'args': vars(args), 'timestamp': kernel_pack.timestamp,
                       'train_metrics': train_metrics, 'eval_metrics': eval_metrics,
                       'extra': extra},
                      fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
