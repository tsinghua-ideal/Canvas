"""
NAS search on ImageNet using Canvas.

Supports searched kernel evaluation on proxy dataset, final discretization and full model training.
Logging and checkpointing throughout the search process.
"""
import inspect
import random 
import math
import itertools
import gc

import numpy as np
import torch
import torch.nn as nn
import timm

from copy import deepcopy
from typing import Union, Callable, List, Dict

import canvas
from canvas import placeholder

import os
import ptflops

from base import dataset, device, log, models, parser, trainer, darts
if __name__ == '__main__':
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()
    logger.info(f'Program arguments: {args}')
    
    # Check available devices and set distributed.
    logger.info(f'Initializing devices ...')
    device.initialize(args)
    assert not args.distributed, 'Search mode does not support distributed training'
    
    # Training utils.
    logger.info(f'Configuring model {args.model} ...')
    model = models.get_model(args, search_mode=True)
    # Get model complexity
    original_macs, original_params = ptflops.get_model_complexity_info(
        model, args.input_size, as_strings=False, print_per_layer_stat=False
    )
    original_macs /= 1e9
    original_params /= 1e6
    
    # Define module mapping
    module_dict = {
        # nn.Conv2d: "conv"
        # ,
        timm.models.resnet.BasicBlock: "resblock"
    }    
    
    # Replace modules with Placeholders
    replaced, not_replaced = darts.replace_module_with_placeholder(model, module_dict)
    logger.info(f'replaced kernel = {replaced}, not replaced kernel = {not_replaced}')
    
    # Calculate complexity after replacement
    placeholder_macs, placeholder_params = ptflops.get_model_complexity_info(
        model, args.input_size,as_strings=False, print_per_layer_stat=False
    )
    placeholder_macs /= 1e9
    placeholder_params /= 1e6
    
    train_loader, eval_loader = dataset.get_loaders(args)
    proxy_train_loader, proxy_eval_loader = dataset.get_loaders(args, proxy=True)
    # Load checkpoint.
    if args.canvas_load_checkpoint:
        logger.info(f'Loading checkpoint from {args.canvas_load_checkpoint}')
        checkpoint = torch.load(args.canvas_load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Set up Canvas randomness seed.
    logger.info(f'Configuring Canvas ...')
    canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)
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
            # canvas.replace(model, pack.module, args.device)
            canvas.replace(model, darts.ReplacedModule, pack, args.device)
            
    # Search.
    
    current_best_score = 0
    train_range = 10
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    kernel_list = []
    for j in range(train_range):
        # Sample a new kernel.
        kernel_list = []
        total_g_macs, total_m_params = placeholder_macs, placeholder_params  
        for i in round_range:
            logger.info('Sampling a new kernel ...')
        
            try:
                kernel_pack = canvas.sample(model,
                                            force_bmm_possibility=args.canvas_bmm_pct,
                                            min_receptive_size=args.canvas_min_receptive_size,
                                            num_primitive_range=(5, 40),
                                            workers=args.canvas_sampling_workers)
                restore_model_params_and_replace([kernel_pack])
                # restore_model_params_and_replace(kernel_pack)
            except RuntimeError as ex:
                # Early exception: out of memory or timeout.
                logger.info(f'Exception: {ex}')
                log.save(args, None, None, None, {'exception': f'{ex}'})
                continue
            # Compute one kernel FLOPs

            one_g_macs, one_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                    as_strings=False, print_per_layer_stat=False)
            one_g_macs, one_m_params = (one_g_macs / 1e9) - placeholder_macs, (one_m_params / 1e6) - placeholder_params
            # one_g_macs, one_m_params = (one_g_macs / 1e9), (one_m_params / 1e6)
            logger.info(f'After the {i + 1}th compression, g_macs = {one_g_macs}, m_params = {one_m_params}')
            logger.info(f'Sampled kernel hash: {hash(kernel_pack)}')
            logger.info(f'MACs: {one_g_macs} G, params: {one_m_params} M')
            macs_not_satisfied = (args.canvas_min_macs > 0 or args.canvas_max_macs > 0) and \
                                (one_g_macs < args.canvas_min_macs or one_g_macs > args.canvas_max_macs)
            params_not_satisfied = (args.canvas_min_params > 0 or args.canvas_max_params > 0) and \
                                (one_m_params < args.canvas_min_params or one_m_params > args.canvas_max_params)
            if macs_not_satisfied or params_not_satisfied:
                logger.info(f'MACs ({args.canvas_min_macs}, {args.canvas_max_macs}) or '
                            f'params ({args.canvas_min_params}, {args.canvas_max_params}) '
                            f'requirements do not satisfy')
                
                continue
            else:
                # Update totals 
                kernel_list.append(kernel_pack)
                total_g_macs += one_g_macs
                total_m_params += one_m_params
                # Strategy one 
                
                # if total_g_macs > og_g_macs * args.compression_rate or total_m_params > og_m_params * args.compression_rate:
                #     kernel_list.pop()
                #     break
                
                # Strategy two
                
                if len(kernel_list) == args.canvas_number_of_kernels:
                    break
        restore_model_params_and_replace(kernel_list)
        final_g_macs, final_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                    as_strings=False, verbose=True, print_per_layer_stat=True)
        final_g_macs, final_m_params = final_g_macs / 1e9, final_m_params / 1e6
        logger.info(f'In the {j + 1}th round, g_macs = {final_g_macs}, m_params = {final_m_params}')
        logger.info(f'expected g macs{total_g_macs}, m_params{total_m_params}')      

        # Evaluate     
        while True:
            try:
                logger.info('Darts evaluating on main dataset ...')
                args.epochs = 5
                train_metrics, eval_metrics = trainer.train(args, model=model,
                                            train_loader=train_loader, eval_loader=eval_loader)
                break
            except RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')
                
        # Discretize the ensembled kernel to get final model
        weight_sharing = True
        darts.get_final_model(model, weightsharing)
        proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
        try:
            logger.info('Training on main dataset ...')
            # model = model.to(args.device)
            args.epochs = 100
            train_metrics, eval_metrics = \
                trainer.train(args, model=model,
                              train_loader=train_loader, eval_loader=eval_loader,
                              search_mode=True)
            score = max([item['top1'] for item in eval_metrics])
            logger.info(f'Solution score: {score}')
            if score > current_best_score:
                current_best_score = score
                logger.info(f'Current best score: {current_best_score}')
                if args.canvas_weight_sharing:
                    try:
                        cpu_clone = deepcopy(model).cpu()
                        logger.info(f'Weight successfully shared')
                    except Exception as ex:
                        logger.warning(f'Failed to make weight shared: {ex}')
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
        # Save into logging directory.
        extra = {'proxy_score': proxy_score, 'g_macs': g_macs, 'm_params': m_params}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, kernel_pack, train_metrics, eval_metrics, extra)