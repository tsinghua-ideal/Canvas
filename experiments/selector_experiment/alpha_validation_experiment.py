import gc
import json
import os
import torch
import time
import random

from copy import deepcopy
from functools import partial

import canvas
from canvas import KernelPack, placeholder

from base import dataset, device, log, models, parser, trainer, darts, darts_trainer, loss, eoi_trainer, proxyless_train
from contextlib import suppress
from base.proxyless import *

if __name__ == '__main__':
    
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()
    
    # Check available devices and set distributed.
    logger.info(f'Initializing devices ...')
    device.initialize(args)

    # Training utils.
    logger.info(f'Configuring model {args.model} ...')
    model = models.get_model(args, search_mode=True)   
    train_loader, eval_loader = dataset.get_loaders(args)

    # Initialization of search.
    cpu_clone = deepcopy(model).cpu()
    output = cpu_clone(torch.randn(1, 3, 224, 224))
    cpu_clone.canvas_cached_placeholders = placeholder.get_placeholders(cpu_clone, None, check_shapes=False)
    
    def restore_model_params_and_replace(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        if pack is not None:
            canvas.replace(model, partial(darts.GumbelParallelKernels, pack), args.device)
            
    def restore_model_params_and_replace_with_single_kernel(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        if pack is not None:
            canvas.replace(model, partial(darts.ParallelKernels, pack), args.device)
            
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    if not os.path.isdir(target_folder):
        logger.info('Given path must be a directory')

    subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    random.seed(time.time())
    random.shuffle(subfolders)
    logger.info(f'Number of kernels: {len(subfolders)}')
    group_number = 0
    
    # Sample
    for start_idx in range(0, len(subfolders), args.canvas_number_of_kernels):
        group_number += 1
        
        # The first and the second kernel are separated by four * args.canvas_number_of_kernels        
        end_idx = start_idx +  args.canvas_number_of_kernels
        if end_idx < len(subfolders):       
            group_folders = subfolders[start_idx:end_idx]
        else:
            group_folders = subfolders[start_idx:]

        folder_names = [f"{folder}" for folder in group_folders]
        logger.info(f'folder_names: {folder_names}')
        group_folders_full_name = [os.path.join(target_folder, f) for f in group_folders]
        
        # Train parallel kernels
        all_eval_metrics, all_train_metrics, drop_eval_metrics = {}, {}, {}
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]  
        module_list = [kernel_pack.module for kernel_pack in kernel_pack_list] 
        restore_model_params_and_replace_with_single_kernel(module_list)   
        exception_info = None  
        try:
            parallel_kernels_train_eval_metrics = trainer.train(args, model=model,
                                        train_loader=train_loader, eval_loader=eval_loader)
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
        
        # Get the real ranking of each kernel
        top1_ranking = {}
        exception_info = None  
        args.epochs = 40
        # 获取子数组
        sample_kernel_list = sort_and_prune(model, kernel_pack_list)
        # Train each kernel pack independently
        for kernel_pack in kernel_pack_list:
            restore_model_params_and_replace_with_single_kernel([kernel_pack.module]) 
            
            try:
                single_kernel_train_eval_metrics = trainer.train(args, model=model,
                                            train_loader=train_loader, eval_loader=eval_loader)
                top1_value = single_kernel_train_eval_metrics["latest_top1"]
                top1_ranking[kernel_pack.name] = top1_value
            except RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')
                continue
        sorted_top1_ranking = sorted(top1_ranking.items(), key=lambda x: x[1])
        
        
        extra = {'magnitude_alphas': parallel_kernels_train_eval_metrics["magnitude_alphas"], 'one_hot_alphas': parallel_kernels_train_eval_metrics["one_hot_alphas"], 'kernel_pack_list': folder_names, 'actual_ranking': sorted_top1_ranking}
        if exception_info:
            extra['exception'] = exception_info
            
        # Make directory (may overwrite).
        if os.path.exists(args.canvas_log_dir):
            assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
        if 'exception' in extra:
            exception_info = extra['exception']
            if 'memory' in exception_info or 'NaN' in exception_info or 'Pruned' in exception_info:
                # Do not record these types.
                continue
            else:
                error_type = 'Error'
            dir_name = f'Canvas_{error_type}_'
        else:
            dir_name = f'Canvas{args.canvas_number_of_kernels}_{group_number}_{os.environ["CUDA_VISIBLE_DEVICES"]}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save args and metrics.
        with open(os.path.join(path, f'{group_number}.json'), 'w') as file:

            json.dump({'args': vars(args),
                    'train_metrics': all_train_metrics, 'eval_metrics': all_eval_metrics,
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)