import time
import json
import logging
import math
import os
import oss2


logging.basicConfig(level=logging.DEBUG)
_exp_logger = logging.getLogger()
_exp_logger.setLevel(logging.INFO)
oss_try_times = 10


def get_logger():
    global _exp_logger
    return _exp_logger


def get_oss_bucket():
    return oss2.Bucket(oss2.Auth('LTAI5tCx79brCnGXxKGTsAst', 'F0IVmA99YzX2x8LWkGrp8WBjVH9qsa'),
                       'oss-cn-hangzhou.aliyuncs.com', 'canvas-imagenet', connect_timeout=5)


def save(args, kernel_pack, train_metrics, eval_metrics, extra):
    if args.canvas_log_dir:
        # Logging info.
        logger = get_logger()
        if kernel_pack:
            logger.info(f'Saving kernel {kernel_pack.name} into {args.canvas_log_dir} ...')
        else:
            logger.info(f'Saving exception into {args.canvas_log_dir} ...')

        # Make directory (may overwrite).
        if os.path.exists(args.canvas_log_dir):
            assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
        if 'exception' in extra:
            exception_info = extra['exception']
            if 'memory' in exception_info or 'NaN' in exception_info:  # Do not record these types.
                return
            else:
                error_type = 'Error'
            dir_name = f'Canvas_{error_type}_'
            dir_name = dir_name + (f'{kernel_pack.name}' if kernel_pack else f'{time.time_ns()}')
        else:
            assert len(eval_metrics) > 0
            max_score = math.floor(max([item['top1'] for item in eval_metrics]) * 100)
            score_str = ('0' * max(0, 5 - len(f'{max_score}'))) + f'{max_score}'
            dir_name = f'Canvas_{score_str}_{kernel_pack.name}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save code, graphviz, args, and results.
        if kernel_pack:
            kernel_pack.save_torch_code(os.path.join(path, kernel_pack.name + '.py'))
            kernel_pack.save_graphviz_code(os.path.join(path, kernel_pack.name + '.dot'))
        else:
            kernel_name = 'exception'
        with open(os.path.join(path, kernel_pack.name + '.json'), 'w') as file:
            json.dump({'args': vars(args), 'timestamp': kernel_pack.timestamp if kernel_pack else None,
                       'train_metrics': train_metrics, 'eval_metrics': eval_metrics,
                       'extra': extra},
                      fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)

        # Save to OSS buckets.
        if args.canvas_oss_bucket:
            logger.info(f'Uploading into OSS bucket {args.canvas_oss_bucket}')
            prefix = args.canvas_oss_bucket + '/' + dir_name + '/'
            for filename in os.listdir(path):
                global oss_try_times
                success = False
                for i in range(oss_try_times):
                    # noinspection PyBroadException
                    try:
                        logger.info(f'Uploading {filename} ...')
                        get_oss_bucket().put_object_from_file(prefix + filename, os.path.join(path, filename))
                        success = True
                        break
                    except Exception as ex:
                        logger.info(f'Failed to upload, try {i + 1} time(s)')
                        continue
                if not success:
                    logger.info(f'Uploading failed for {oss_try_times} time(s)')
