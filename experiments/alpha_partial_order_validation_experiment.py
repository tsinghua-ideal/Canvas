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
                canvas.replace(model, partial(ParallelKernels_Test, pack), args.device)
            else:
                canvas.replace(model, pack, args.device)
        else:
            NotImplementedError()      
            
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    single_result_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/validation_experiments/single_with_compact_van_new"
    subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    seed = time.time()
    random.seed(seed)
    random.shuffle(subfolders)
    logger.info(f'Number of kernels: {len(subfolders)}')
    group_number = 1
    start_idx = 0
    cur_idx = start_idx
    group_folders = []
    
    # Sample
    while start_idx < len(subfolders): 
        while len(group_folders) < args.canvas_number_of_kernels and cur_idx < len(subfolders):
            kernel_pack = KernelPack.load_from_dir(os.path.join(target_folder, subfolders[cur_idx]))
            target_file = os.path.join(single_result_folder, f'{kernel_pack.name}/metrics.json')
            if os.path.exists(target_file):
                group_folders.append(subfolders[cur_idx])
                if len(group_folders) == int(0.5 * args.canvas_number_of_kernels):
                    new_start_idx = cur_idx + 1
                cur_idx += 1
            else:
                cur_idx += 1
        if cur_idx == len(subfolders):
            break
        end_idx = cur_idx + 1
        cur_idx = new_start_idx
        folder_names = [f"{folder}" for folder in group_folders]
        logger.info(f'folder_names: {folder_names}')
        group_folders_full_name = [os.path.join(target_folder, f) for f in group_folders]
        
        # Train parallel kernels
        all_eval_metrics = {}
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]  
        module_list = [kernel_pack.module for kernel_pack in kernel_pack_list] 
        restore_model_params_and_replace(module_list)  
        group_folders = [] 
        try:
            parallel_kernels_train_eval_metrics = proxyless_trainer.train(args, model=model,
                                        train_loader=train_loader, valid_loader=valid_loader, eval_loader=eval_loader)
            if parallel_kernels_train_eval_metrics['all_eval_metrics'][-1]['top1'] < 70:
                continue
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
        
        # Get the real ranking of each kernel
        top1_ranking, sum_sorted_top1_ranking = {}, {}
        
        # Get other scores
        multiplication_probs_log = get_multiplication_of_magnitude_probs_with_1D(model)
        corresponding_scores, sample_kernel_list = sort_and_prune(alpha_list=multiplication_probs_log.tolist(), kernel_list=kernel_pack_list)
        
        exception_info = None  
        
        for kernel_pack in sample_kernel_list:
            target_file = os.path.join(single_result_folder, f'{kernel_pack.name}/metrics.json')
            if os.path.exists(target_file):
                logger.info(f'{kernel_pack.name} has been trained')
                with open(target_file, "r") as json_file:
                    data = json.load(json_file)
                    assert 'top1_value' in data['extra']
                    top1_ranking[kernel_pack.name] = data['extra']['top1_value']
            else:
                logger.info(f'{kernel_pack.name} has not been trained')
                top1_ranking[kernel_pack.name] = 0
        sorted_top1_ranking = sorted(top1_ranking.items(), key=lambda x: x[1], reverse=True)
        sorted_top1_ranking_dict = {}
        for i, (name, value) in enumerate(sorted_top1_ranking):
            sorted_top1_ranking_dict[name] = (value, i + 1)
        compare_dict = {}  
        for i, kernel_pack in enumerate(sample_kernel_list):
            compare_dict[i + 1] = (kernel_pack.name, sorted_top1_ranking_dict[kernel_pack.name][1])
        group_number += 1 
        
        # 
        compare_dict_full = {}
        for epoch in range(args.warmup_epochs + 1, args.epochs, 5):
            compare_dict_i, top1_ranking_i = {}, {}
            corresponding_scores, sample_kernel_list = sort_and_prune(alpha_list=parallel_kernels_train_eval_metrics['magnitude_alphas'][epoch], kernel_list=kernel_pack_list)
            for kernel_pack in sample_kernel_list:
                target_file = os.path.join(single_result_folder, f'{kernel_pack.name}/metrics.json')
                if os.path.exists(target_file):
                    logger.info(f'{kernel_pack.name} has been trained')
                    with open(target_file, "r") as json_file:
                        data = json.load(json_file)
                        assert 'top1_value' in data['extra']
                        top1_ranking_i[kernel_pack.name] = data['extra']['top1_value']
                else:
                    logger.info(f'{kernel_pack.name} has not been trained')
                    top1_ranking_i[kernel_pack.name] = 0
            sorted_top1_ranking_i = sorted(top1_ranking_i.items(), key=lambda x: x[1], reverse=True)
            sorted_top1_ranking_dict = {}
            for j, (name, value) in enumerate(sorted_top1_ranking_i):
                sorted_top1_ranking_dict[name] = (value, j + 1)
            
            for k, kernel_pack in enumerate(sample_kernel_list):
                compare_dict_i[k + 1] = (kernel_pack.name, sorted_top1_ranking_dict[kernel_pack.name][1])
            compare_dict_full[f'epoch_{epoch}'] = compare_dict_i
            
        extra = {'kernel_pack_list': folder_names, 'multiplication_probs_log': multiplication_probs_log.tolist(), \
                 'sorted_top1_ranking': sorted_top1_ranking, 'compare_dict': compare_dict, \
                 'magnitude_alphas': parallel_kernels_train_eval_metrics['magnitude_alphas'], 'compare_dict_full': compare_dict_full}
        
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
            dir_name = f'Canvas_{seed}_{start_idx}_to_{end_idx}_{args.canvas_number_of_kernels}_{group_number}_machine_{socket.gethostname()}_GPU_{os.environ["CUDA_VISIBLE_DEVICES"]}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save args and metrics.
        with open(os.path.join(path, 'metrics.json'), 'w') as file:
            json.dump({'args': vars(args), 'eval_metrics': parallel_kernels_train_eval_metrics['all_eval_metrics'],
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
        
        # Update start_idx.
        start_idx = new_start_idx