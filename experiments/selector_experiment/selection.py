import fcntl
import gc
import json
import os
import pickle
import torch
import time
import random

from copy import deepcopy
from functools import partial

import canvas
from canvas import KernelPack, placeholder

from base import dataset, device, log, models, parser, proxyless_trainer
from experiments.base.models.proxyless import *

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
            canvas.replace(model, partial(ParallelKernels, pack), args.device)

    
    
    # Seed random number generator with current time and shuffle the current subfolders
    random.seed(time.time())
    group_number = 0
    oom_count = 0
    FILE = f'{args.target_folder}/my_container.pkl'
    
    # Sample
    while True:
        group_number += 1
        file = open(FILE, 'rb+')
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        print('acquire lock') 
        my_container = pickle.load(file)
        end_idx = my_container.start_idx + args.canvas_number_of_kernels
        if end_idx < len(my_container.cur_subfolders):       
            group_folders = my_container.cur_subfolders[my_container.start_idx:end_idx]
            my_container.start_idx += args.canvas_number_of_kernels  
        else:
            group_folders = my_container.cur_subfolders[my_container.start_idx:]
            if len(my_container.new_subfolders) < args.canvas_number_of_kernels:
                break
            else:
                
                # Shuffle next-round's elements
                random.shuffle(my_container.new_subfolders)
                my_container.start_idx = 0
                my_container.cur_subfolders = my_container.new_subfolders
                my_container.new_subfolders = []
                
        file.seek(0)  
        pickle.dump(my_container, file)
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        file.close()
        time.sleep(2)
        print('release lock') 
        
        folder_names = [f"{folder}" for folder in group_folders]
        logger.info(f'folder_names: {folder_names}')
        group_folders_full_name = [os.path.join(args.target_folder, f) for f in group_folders]
        
        # Train parallel kernels
        all_eval_metrics, all_train_metrics = {}, {}
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]  
        module_list = [kernel_pack.module for kernel_pack in kernel_pack_list] 
        restore_model_params_and_replace(module_list)   
        exception_info = None  
        try:
            parallel_kernels_train_eval_metrics = proxyless_trainer.train(args, model=model,
                                        train_loader=train_loader, eval_loader=eval_loader)
            file = open(FILE, 'rb+')
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            my_container = pickle.load(file)
            my_container.new_subfolders.append(sort_and_prune(get_sum_of_magnitude_scores_with_1D(model), group_folders))
            file.seek(0)
            pickle.dump(my_container, file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
            
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            file = open(FILE, 'rb+')
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            my_container = pickle.load(file)
            my_container.cur_subfolders.append(group_folders)
            file.seek(0)
            pickle.dump(my_container, file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()
                
            oom_count += 1
            continue
        
        extra = {'kernel_pack_list': folder_names}
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
            
    # Make directory (may overwrite).
    if os.path.exists(args.canvas_log_dir):
        assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
    if 'exception' in extra:
        exception_info = extra['exception']
        dir_name = f'Canvas_{error_type}_'
    else:
        dir_name = f'Final_Candidates'
    path = os.path.join(args.canvas_log_dir, dir_name)
    if os.path.exists(path):
        logger.info('Overwriting results ...')
    os.makedirs(path, exist_ok=True)
    args.oom_count = oom_count
    
    # Save args and metrics.
    with open(os.path.join(path, f'{group_number}_{os.environ["CUDA_VISIBLE_DEVICES"]}.json'), 'w') as file:
        json.dump({'args': vars(args)},
                fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)