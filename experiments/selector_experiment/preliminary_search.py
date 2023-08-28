"""
Kernel search on CV using Canvas.

Supports searched kernel evaluation on proxy dataset, final discretization and full model training.
Logging and checkpointing throughout the search process.
"""
import random 
import itertools
import gc

import torch
import torch.nn as nn

from copy import deepcopy
from functools import partial
import canvas
from canvas import placeholder

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
    model, original_macs, original_params = models.get_model(args, search_mode=True)
    
    # Replace modules with Placeholders
    if args.canvas_van_tiny == False:
        
        # Define module mapping
        module_dict = {
            nn.Conv2d: "conv"
            # ,
            # timm.models.resnet.BasicBlock: "resblock"
        } 
        
        replaced, not_replaced = darts.replace_module_with_placeholder(model, module_dict)
        logger.info(f'replaced kernel = {replaced}, not replaced kernel = {not_replaced}')
    
        # Calculate complexity after replacement
        placeholder_macs, placeholder_params = ptflops.get_model_complexity_info(
            model, args.input_size,as_strings=False, print_per_layer_stat=False
        )
        placeholder_macs /= 1e9
        placeholder_params /= 1e6
    else:
        placeholder_macs, placeholder_params = original_macs, original_params
        
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
            canvas.replace(model, partial(darts.ParallelKernels, pack), args.device)
            
    # Search.
    current_best_score = 0
    train_range = 10
    round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
    logger.info(f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)')
    passed, failed = 0, 0
      
    # Sample a new kernel.
    args.epochs = 1
    total_g_macs, total_m_params = placeholder_macs, placeholder_params  
    for j in round_range:
        if passed == 2000:
            break
        logger.info('Sampling a new kernel ...')
        try:
            kernel_pack = canvas.sample(model,
                                        force_bmm_possibility=args.canvas_bmm_pct,
                                        min_receptive_size=args.canvas_min_receptive_size,
                                        num_primitive_range=(5, 40),
                                        workers=args.canvas_sampling_workers)
            restore_model_params_and_replace([kernel_pack.module])
        except RuntimeError as ex:
            # Early exception: out of memory or timeout.
            logger.info(f'Exception: {ex}')
            log.save(args, None, None, None, None, {'exception': f'{ex}'})
            continue
        
        # Compute one kernel FLOPs
        one_g_macs, one_m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                as_strings=False, print_per_layer_stat=False)
        one_g_macs, one_m_params = (one_g_macs / 1e9) - placeholder_macs, (one_m_params / 1e6) - placeholder_params
        # one_g_macs, one_m_params = (one_g_macs / 1e9), (one_m_params / 1e6)
        logger.info(f'After the {j + 1}th compression, g_macs = {one_g_macs}, m_params = {one_m_params}')

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
            
        # Evaluate     
        try:
            logger.info('Darts evaluating on main dataset ...')

            train_metrics, eval_metrics = trainer.train(args, model=model,
                                        train_loader=train_loader, eval_loader=eval_loader, evaluate = True)
    
            # 获取最近两次的'top1'值
            latest_top1_values = [metrics['top1'] for metrics in eval_metrics[-2:]]

            # 计算最近两次'top1'的平均值
            average_top1 = sum(latest_top1_values) / len(latest_top1_values)

            # If worse than backbone VAB[3, 3, 5, 2], then pass TODO 
            if average_top1 < 49.65:
                failed += 1
                continue
            else:
                passed += 1
            logger.info(f'passed:{passed}, failed{failed}')   
            
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')
            continue
           
        # Save into logging directory.
        exception_info = None
        extra = {}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, [kernel_pack], kernel_pack, train_metrics, eval_metrics, extra)
