import gc
import json
import os
import torch
import random
import time

from copy import deepcopy
from functools import partial

import canvas
from canvas import KernelPack

from base import dataset, device, log, models, parser, trainer
from base.models.proxyless import *

    
if __name__ == '__main__':
    
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()
    
    # Check available devices and set distributed.
    logger.info(f'Initializing devices ...')
    device.initialize(args)
    torch.backends.cudnn.enabled = True
    
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
                canvas.replace(model, partial(ParallelKernels, pack), args.device)
            else:
                canvas.replace(model, pack, args.device)
        else:
            NotImplementedError()      
            
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    single_result_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/validation_experiments/single_cifar100"
    subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    seed = time.time()
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
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]  
        
        # Get the real ranking of each kernel
        top1_ranking, sum_sorted_top1_ranking = {}, {}
        exception_info = None  
        
        # Train each kernel pack independently
        for kernel_pack in kernel_pack_list:
            path = os.path.join(single_result_folder, f'{kernel_pack.name}')
            if os.path.exists(path):
                logger.info(f'{kernel_pack.name} has been trained')
                continue
            restore_model_params_and_replace(kernel_pack.module) 
            try:
                single_kernel_train_eval_metrics = trainer.train(args=args, model=model, \
                                        train_loader=train_loader, eval_loader=eval_loader)
                top1_value = single_kernel_train_eval_metrics['best_metric']
                top1_ranking[kernel_pack.name] = top1_value
            except   RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')
                continue
            extra = {'top1_value': top1_value, 'best_epoch': single_kernel_train_eval_metrics['best_epoch']}
            if exception_info:
                extra['exception'] = exception_info


            # Make directory (may overwrite).
            if 'exception' in extra:
                exception_info = extra['exception']
                if 'memory' in exception_info or 'NaN' in exception_info or 'Pruned' in exception_info:
                    # Do not record these types.
                    continue
                else:
                    error_type = 'Error'
                dir_name = f'Canvas_{error_type}_'
            else:
                dir_name = f'{kernel_pack.name}'
            if os.path.exists(path):
                logger.info('Overwriting results ...')
            os.makedirs(os.path.join(single_result_folder, f'{dir_name}'), exist_ok=True)

            # Save args and metrics.
            with open(os.path.join(path, 'metrics.json'), 'w') as file:
                json.dump({'args': vars(args), 'extra': extra},
                        fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
        sum_sorted_top1_ranking = sorted(top1_ranking.items(), key=lambda x: x[1])
        group_number += 1 

        