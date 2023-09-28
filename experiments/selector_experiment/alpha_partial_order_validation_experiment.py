import gc
import json
import os
import torch
import time
import random
import socket

from copy import deepcopy
from functools import partial

import canvas
from canvas import KernelPack, placeholder

from base import dataset, device, log, models, parser, trainer, proxyless_trainer
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
    
    def restore_model_params_and_replace(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        if pack is not None:
            if isinstance(pack, list):
                canvas.replace(model, partial(ProxylessParallelKernels, pack), args.device)
            else:
                canvas.replace(model, pack, args.device)
        else:
            NotImplementedError()      
            
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
        end_idx = start_idx + args.canvas_number_of_kernels
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
        restore_model_params_and_replace(module_list)   
        try:
            parallel_kernels_train_eval_metrics = proxyless_trainer.train(args, model=model,
                                        train_loader=train_loader, eval_loader=eval_loader)
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
        
        # Get the real ranking of each kernel
        top1_ranking, sum_sorted_top1_ranking = {}, {}
        exception_info = None  
        args.epochs = 50
        
        # Get the subset of kernels according to the ranking of their scores to train independently
        sum_sample_kernel_list, sum_corresponding_scores = sort_and_prune(alpha_list=get_sum_of_magnitude_scores_with_1D(model), kernel_list=kernel_pack_list)
        
        # Train each kernel pack independently
        torch.backends.cudnn.enabled = True
        for kernel_pack in sum_sample_kernel_list:
            restore_model_params_and_replace(kernel_pack.module) 
            try:
                single_kernel_train_eval_metrics = trainer.train(args, model=model,
                                            train_loader=train_loader, eval_loader=eval_loader)
                top1_value = single_kernel_train_eval_metrics["latest_top1"]
                top1_ranking[kernel_pack.name] = top1_value
            except   RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')
                continue
        sum_sorted_top1_ranking = sorted(top1_ranking.items(), key=lambda x: x[1])
        
        torch.backends.cudnn.enabled = False
        extra = {'magnitude_alphas': parallel_kernels_train_eval_metrics["magnitude_alphas"],'kernel_pack_list': folder_names, 'sum_sorted_top1_ranking': sum_sorted_top1_ranking, 'sum_kernel_used': [kernel_pack.name for kernel_pack in sum_sample_kernel_list], 'sum_corresponding_scores': sum_corresponding_scores}
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
            dir_name = f'Canvas_{args.canvas_number_of_kernels}_{group_number}_machine_{socket.gethostname()}_GPU_{os.environ["CUDA_VISIBLE_DEVICES"]}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save args and metrics.
        with open(os.path.join(path, 'metrics.json'), 'w') as file:
            json.dump({'args': vars(args),
                    'train_metrics': all_train_metrics, 'eval_metrics': all_eval_metrics,
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)