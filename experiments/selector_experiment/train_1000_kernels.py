import gc
import os
import torch
import math
import json
from copy import deepcopy
from functools import partial
import canvas
from canvas import placeholder, KernelPack
from base import dataset, device, log, models, parser, trainer, darts

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
            canvas.replace(model, partial(darts.ParallelKernels, pack), args.device)
            
    target_folder = "/scorpio/home/shenao/myProject/Canvas/experiments/collections/preliminary_kernels_selected"  
    if not os.path.isdir(target_folder):
        logger.info('Given path must be a directory')

    subfolders = [f.name for f in os.scandir(target_folder) if f.is_dir()]
    subfolders.sort()
    logger.info(f'Number of kernels: {len(subfolders)}')
    group_number = 0
    
    # Sample
    for start_idx in range(3, len(subfolders), args.canvas_number_of_kernels):
        group_number += 1
        end_idx = start_idx + args.canvas_number_of_kernels       
        group_folders = subfolders[start_idx:end_idx]
        # If less than the kernels needed, use the ones at the beginning
        if len(group_folders) < args.canvas_number_of_kernels:
            group_folders.extend(subfolders[:args.canvas_number_of_kernels - len(group_folders)])

        folder_names = [f"{folder}" for folder in group_folders]
        logger.info(f'{start_idx},{end_idx},' + ",".join(folder_names))
        group_folders_full_name = [os.path.join(target_folder, f) for f in group_folders]
        
        # Train
        kernel_pack_list = [KernelPack.load_from_dir(dir) for dir in group_folders_full_name]   
        restore_model_params_and_replace([kernel_pack.module for kernel_pack in kernel_pack_list])   

        exception_info = None
        # Evaluate     
        try:
            all_train_eval_metrics = trainer.train(args, model=model,
                                        train_loader=train_loader, eval_loader=eval_loader, evaluate = True)
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
        extra = {'magnitude_alphas': all_train_eval_metrics["magnitude_alphas"], 'one_hot_alphas': all_train_eval_metrics["one_hot_alphas"],'kernel_pack_list': folder_names}
        
        if exception_info:
            extra['exception'] = exception_info
        eval_metrics = all_train_eval_metrics["all_eval_metrics"]  
        train_metrics = all_train_eval_metrics["all_train_metrics"]   
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
            assert len(eval_metrics) > 0
            max_score = math.floor(max([item['top1'] for item in eval_metrics]) * 100)
            score_str = ('0' * max(0, 5 - len(f'{max_score}'))) + f'{max_score}'
            dir_name = f'Canvas_{score_str}_{group_number}'
        path = os.path.join(args.canvas_log_dir, dir_name)
        if os.path.exists(path):
            logger.info('Overwriting results ...')
        os.makedirs(path, exist_ok=True)

        # Save args, and results.
        with open(os.path.join(path, f'{group_number}.json'), 'w') as file:
            json.dump({'args': vars(args),
                    'train_metrics': train_metrics, 'eval_metrics': eval_metrics,
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)