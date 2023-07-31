
import gc
import itertools
import torch
import os
import torch.nn as nn
import ptflops
import numpy as np
import math
import canvas
import random
import timm
import inspect
from canvas import placeholder
from typing import Union, Callable, List, Dict
from copy import deepcopy
from base import dataset, device, log, models, parser, trainer, darts

from thop import profile
if __name__ == '__main__':
    # Get arguments.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
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
    # for name, param in model.named_parameters():
    #     print(name, param.device)
    g_macs, m_params = 0, 0
    # model = Kernel_14187586262933953542(3, 244, 244)
    og_g_macs, og_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                 as_strings=False, print_per_layer_stat=False)
    og_g_macs, og_m_params = og_g_macs / 1e9, og_m_params / 1e6
    inputs = torch.randn(1, 3, 224, 224).to(args.device)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops)
    print('params: ', params)
    print(f"The model given by the user, g_macs = {og_g_macs}, m_params = {og_m_params}")
    # print(model)
    module_dict = {
        # nn.Conv2d: "conv"
        # ,
        timm.models.resnet.BasicBlock: "resblock"
    }    
    replaced, not_replaced = darts.replace_module_with_placeholder(model, module_dict)
    # print("Model  (after replacement):")
    # print(model)
    print(f"replaced kernel = {replaced}, not replaced kernel = {not_replaced}")
    ph_g_macs, ph_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                    as_strings=False, verbose=True, print_per_layer_stat=True)
    ph_g_macs, ph_m_params = ph_g_macs / 1e9, ph_m_params / 1e6
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
    # model = torch.nn.DataParallel(model)
    def restore_model_params_and_replace(pack=None):
        global model
        model = None
        gc.collect()
        model = deepcopy(cpu_clone).to(args.device)
        # for name, param in model.named_parameters():
        #         print(name, param.device)
        if pack is not None:
            # canvas.replace(model, pack.module, args.device)
            canvas.replace(model, darts.replaced_module, pack, args.device)
            
    # Search.
    
    current_best_score = 0
    train_range = 10
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    kernel_list = []
    for j in range(train_range):
        # Sample a new kernel.
        kernel_list = []
        total_g_macs, total_m_params = ph_g_macs, ph_m_params
        
        
        for i in round_range:
            logger.info('Sampling a new kernel ...')
        
            try:
                
                # g_macs, m_params = ptflops.get_model_complexity_info(model, args.input_size,
                #                                                     as_strings=False, verbose=True, print_per_layer_stat=False)
                # g_macs, m_params = g_macs / 1e9, m_params / 1e6
                # for name, param in model.named_parameters():
                #     print(name, param.device)
                # print(f"Before the {i + 1}th compression, g_macs = {g_macs}, m_params = {m_params}")
                # inputs = torch.randn(1, 3, 224, 224).to(args.device)
                # flops, params = profile(model, (inputs,))
                # print('flops: ', flops)
                # print('params: ', params)
                kernel_pack = canvas.sample(model,
                                            force_bmm_possibility=args.canvas_bmm_pct,
                                            min_receptive_size=args.canvas_min_receptive_size,
                                            num_primitive_range=(5, 40),
                                            workers=args.canvas_sampling_workers)
                # print(f"model:{model}")
                # print(kernel_pack.module)
                # TEST
                restore_model_params_and_replace([kernel_pack])
                # restore_model_params_and_replace(kernel_pack)
            except RuntimeError as ex:
                # Early exception: out of memory or timeout.
                logger.info(f'Exception: {ex}')
                log.save(args, None, None, None, {'exception': f'{ex}'})
                continue
            # Calculate the macs and params of the kernel itself(minus the one of skeleton)

            # one_g_macs, one_m_params = ptflops.get_model_complexity_info(model, args.input_size,
            #                                                         as_strings=False, verbose=True, print_per_layer_stat=True)
            one_g_macs, one_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                    as_strings=False, print_per_layer_stat=False)
            one_g_macs, one_m_params = (one_g_macs / 1e9) - ph_g_macs, (one_m_params / 1e6) - ph_m_params
            # one_g_macs, one_m_params = (one_g_macs / 1e9), (one_m_params / 1e6)
            print(f"After the {i + 1}th compression, g_macs = {one_g_macs}, m_params = {one_m_params}")
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
                kernel_list.append(kernel_pack)
                total_g_macs.add(one_g_macs)
                total_m_params.add(one_m_params)
                # Strategy one 
                
                # if total_g_macs > og_g_macs * 0.5 or total_m_params > og_m_params * 0.5:
                #     kernel_list.pop()
                #     break
                
                # Strategy two
                if len(kernel_list) == 2:
                    break
        restore_model_params_and_replace(kernel_list)
        final_g_macs, final_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                    as_strings=False, verbose=True, print_per_layer_stat=False)
        final_g_macs, final_m_params = final_g_macs / 1e9, final_m_params / 1e6
        print(f"In the {j + 1}th round, g_macs = {final_g_macs}, m_params = {final_m_params}")  
        print(f"expected g macs{total_g_macs}, m_params{total_m_params}")          

        # Evaluate
        while True:
            try:
                # torch.cuda.empty_cache()
                # gc.collect()
                # torch.cuda.empty_cache()
                # restore_model_params_and_replace(kernel_pack)
                logger.info('Darts evaluating on main dataset ...')
                # torch.cuda.empty_cache()
                train_metrics = trainer.train(args, model=model,
                                train_loader=train_loader, eval_loader=eval_loader, darts_eval=True,
                                search_mode=True)
                break
            except RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')
        
        
        
        # Discretize the ensembled kernel to get final model
        weightsharing = True
        darts.get_final_model(model, weightsharing)
        proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
        try:
            # if proxy_train_loader and proxy_eval_loader:
            #     logger.info('Training on proxy dataset ...')
            #     _, proxy_eval_metrics = \
            #         trainer.train(args, model=model,
            #                       train_loader=proxy_train_loader, eval_loader=proxy_eval_loader,
            #                       search_mode=True, proxy_mode=True)
            #     assert len(proxy_eval_metrics) > 0
            #     best_epoch = 0
            #     for e in range(1, len(proxy_eval_metrics)):
            #         if proxy_eval_metrics[e]['top1'] > proxy_eval_metrics[best_epoch]['top1']:
            #             best_epoch = e
            #     proxy_score = proxy_eval_metrics[best_epoch]['top1']
            #     kernel_scales = proxy_eval_metrics[best_epoch]['kernel_scales']
            #     restore_model_params_and_replace(kernel_pack)
            #     logger.info(f'Proxy dataset score: {proxy_score}')
            #     if proxy_score < args.canvas_proxy_threshold:
            #         logger.info(f'Under proxy threshold {args.canvas_proxy_threshold}, skip main dataset training')
            #         continue
            #     if len(kernel_scales) > 0 and args.canvas_proxy_kernel_scale_limit > 0:
            #         g_mean = np.exp(np.log(kernel_scales).mean())
            #         if g_mean < args.canvas_proxy_kernel_scale_limit or \
            #            g_mean > 1 / args.canvas_proxy_kernel_scale_limit:
            #             logger.info(f'Breaking proxy scale limit {args.canvas_proxy_kernel_scale_limit} '
            #                         f'(gmean={g_mean}), '
            #                         f'skip main dataset training')
            #             continue
            logger.info('Training on main dataset ...')
            # model = model.to(args.device)
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
