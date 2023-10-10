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
from canvas import KernelPack

from base import dataset, device, log, models, parser, proxyless_trainer
from base.models.proxyless import *

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
    train_loader, valid_loader, eval_loader = dataset.get_loaders(args)

    # Initialization of search.
    cpu_clone = deepcopy(model).cpu()

    def restore_model_params_and_replace(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        if pack is not None:
            if isinstance(pack, list):
                canvas.replace(model, partial(ParallelKernels, pack), args.device)
            else:
                canvas.replace(model, pack, args.device)
        else:
            NotImplementedError()      
            
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    single_result_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/validation_experiments/single"
    subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    seed = 42
    random.seed(seed)
    random.shuffle(subfolders)
    logger.info(f'Number of kernels: {len(subfolders)}')
    group_number = 1
    
    # Sample
    for start_idx in range(0, len(subfolders), args.canvas_number_of_kernels): 
        end_idx = start_idx + args.canvas_number_of_kernels
        if end_idx < len(subfolders):       
            group_folders = subfolders[start_idx:end_idx]
        else:
            group_folders = subfolders[start_idx:]

        folder_names = [f"{folder}" for folder in group_folders]
        logger.info(f'folder_names: {folder_names}')
        group_folders_full_name = [os.path.join(target_folder, f) for f in group_folders]
        
        # Train parallel kernels
        all_eval_metrics = {}
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]  
        module_list = [kernel_pack.module for kernel_pack in kernel_pack_list] 
        restore_model_params_and_replace(module_list)   
        try:
            parallel_kernels_train_eval_metrics = proxyless_trainer.train(args, model=model,
                                        train_loader=train_loader, valid_loader=valid_loader, eval_loader=eval_loader)
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
        
        # Get the real ranking of each kernel
        top1_ranking, sum_sorted_top1_ranking = {}, {}
        
        # Get the kernel result according to the score
        sum_sample_kernel_list, sum_corresponding_scores = sort_and_prune(alpha_list=get_sum_of_magnitude_scores_with_1D(model), kernel_list=kernel_pack_list)
        
        # Get other scores
        magnitude_probs_sum = get_sum_of_magnitude_probs_with_1D(model).tolist()
        multiplication_probs_log = get_multiplication_of_magnitude_probs_with_1D(model).tolist()
        exception_info = None  
        
        for kernel_pack in sum_sample_kernel_list:
            target_file = os.path.join(single_result_folder, f'{kernel_pack.name}/metrics.json')
            if os.path.exists(target_file):
                with open(target_file, "r") as json_file:
                    data = json.load(json_file)
                    assert 'top1_value' in data['extra']
                    top1_ranking[kernel_pack.name] = data['extra']['top1_value']
            else:
                top1_ranking[kernel_pack.name] = 0
        sum_sorted_top1_ranking = sorted(top1_ranking.items(), key=lambda x: x[1])
        group_number += 1 
        extra = {'kernel_pack_list': folder_names, 'magnitude_scores_sum': parallel_kernels_train_eval_metrics['magnitude_alphas'], \
                 'magnitude_probs_sum': magnitude_probs_sum, 'multiplication_probs_log': multiplication_probs_log, \
                 'sum_sorted_top1_ranking': sum_sorted_top1_ranking}
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
            dir_name = f'Canvas_{args.canvas_number_of_kernels}_{group_number}_machine_{socket.gethostname()}_GPU_{os.environ["CUDA_VISIBLE_DEVICES"]}_{seed}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save args and metrics.
        with open(os.path.join(path, 'metrics.json'), 'w') as file:
            json.dump({'args': vars(args), 'eval_metrics': parallel_kernels_train_eval_metrics['all_eval_metrics'],
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)