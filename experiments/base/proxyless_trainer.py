import math
import os
import time
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn.functional as F
import torch.profiler

from timm.models import model_parameters, resume_checkpoint, safe_model_name
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad, distribute_bn, reduce_tensor
from timm.utils import NativeScaler, ApexScaler, CheckpointSaver, get_outdir, update_summary
from torch.utils.tensorboard import SummaryWriter

from . import log, loss, optim, sche, darts
from .proxyless import *

def get_update_schedule(args, nBatch):
    schedule = {}
    for i in range(nBatch):
        if (i + 1) % args.grad_update_arch_param_every == 0:
            schedule[i] = args.grad_update_steps
    return schedule


def train_one_epoch(args, epoch, model, train_loader, eval_loader,
                    train_loss_func, eval_loss_func, w_optimizer, arch_optimizer, lr_scheduler,
                    amp_autocast, loss_scaler, logger, writer=None, pruning_milestones=None, profiler=None):
    # Second order optimizer.
    second_order = hasattr(w_optimizer, 'is_second_order') and w_optimizer.is_second_order
    
    # Proxy
    nBatch = len(train_loader)
    arch_update_schedule = get_update_schedule(args, nBatch)
    num_updates = epoch * nBatch
    
    # Meters.
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()

    # Switch to training mode.
    model.train()

    # Iterate over this epoch.
    last_idx = nBatch - 1
    for batch_idx, (image, target)in enumerate(train_loader):
        data_time_m.update(time.time() - end) 
        image, target = image.to(args.device), target.to(args.device)
        
        # Random sample binary gates
        reset_binary_gates_test(model)  
        
        # Remove unused module for speedup
        unused_modules_off(model)  

        with amp_autocast():
            output = model(image)
            loss_value = train_loss_func(output, target)
        if not args.distributed:
            losses_m.update(loss_value.item(), image.size(0))
        # zero grads of weight_param, arch_param & binary_param
        model.zero_grad() 
         
        if loss_scaler is not None:
            loss_scaler(
                loss_value, w_optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss_value.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            w_optimizer.step()
            
        if args.needs_profiler and epoch >= 5:
            profiler.step()
            
        # Back to normal mode
        unused_modules_back(model)
        
        # Architecture parameter updates
        if epoch > args.warmup_epochs:
            
            # Update architecture parameters according to update_schedule
            try:
                image_eval, target_eval = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(eval_loader)
                image_eval, target_eval = next(valid_queue_iter)
            for j in range(arch_update_schedule.get(batch_idx, 0)):
                
                # Switch to train mode
                model.train()
                
                # Mix edge mode
                ProxylessParallelKernels.MODE = args.grad_binary_mode
                
                image_eval, target_eval = image_eval.to(args.device), target_eval.to(args.device)

                # Random sample binary gates
                reset_binary_gates_test(model)  
                
                # Remove unused module for speedup
                unused_modules_off(model)  
                with amp_autocast():
                    output_eval = model(image_eval)
                if isinstance(output_eval, (tuple, list)):
                    output_eval = output_eval[0]
                    
                # Augmentation reduction.
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output_eval = output_eval.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target_eval = target_eval[0:target_eval.size(0):reduce_factor]
                    
                # Loss
                loss = eval_loss_func(output_eval, target_eval)
                
                # Compute gradient
                model.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                
                # Set architecture parameter gradients
                set_arch_param_grad(model)
                arch_optimizer.step()
                if ProxylessParallelKernels.MODE == 'two':
                    rescale_updated_arch_param(model)
                    
                # Back to normal mode
                unused_modules_back(model)
                ProxylessParallelKernels.MODE = None           
        
        # Sync.
        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        # Update scheduler step.
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        # Update time.
        end = time.time()
   
        # Logging.
        if batch_idx == last_idx or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in w_optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            # Log the parameters of the canvas kernels using tensorboard
            if writer:
                writer.add_scalar('Training/Loss', losses_m.avg, epoch)
                writer.add_scalar('Training/Learning Rate', 
                            lr, epoch)
                
            if args.distributed:
                reduced_loss = reduce_tensor(loss_value.data, args.world_size)
                losses_m.update(reduced_loss.item(), image.size(0))

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(train_loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=image.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=image.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))
                progress = '{:.0f}'.format(100. * batch_idx / last_idx)
                if pruning_milestones and progress in pruning_milestones and \
                        losses_m.avg > pruning_milestones[progress]:
                    raise RuntimeError(f'Pruned by milestone settings at progress {progress}%')

            if math.isnan(losses_m.avg):
                break

    if hasattr(w_optimizer, 'sync_lookahead'):
        w_optimizer.sync_lookahead()
        
    metrics = OrderedDict([('loss', losses_m.avg)])
    return metrics


def validate(args, model, eval_loader, loss_func, amp_autocast, logger):
    model.eval()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    end = time.time()
    last_idx = len(eval_loader) - 1
    
    # set chosen op active
    set_chosen_module_active(model)
    
    # remove unused modules
    unused_modules_off(model)
    
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(eval_loader):
            with amp_autocast():
                output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
                
            # Augmentation reduction.
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            
            loss_value = loss_func(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Distributed data recorded reduction.
            if args.distributed:
                reduced_loss = reduce_tensor(loss_value.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss_value.data            
            
            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), image.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            if math.isnan(losses_m.avg) and args.forbid_eval_nan:
                break

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging.
            if args.local_rank == 0 and (batch_idx == last_idx or batch_idx % args.log_interval == 0):
                logger.info(
                    'Validate: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
                
    unused_modules_back(model)
    
    # Record kernel scales.
    kernel_scales = []
    if hasattr(model, 'kernel_scales'):
        kernel_scales = model.kernel_scales()
        logger.info(f'Kernel scales: {kernel_scales}')

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg),
                        ('top5', top5_m.avg), ('kernel_scales', kernel_scales)])


def train(args, model, train_loader, eval_loader, search_mode: bool = False, proxy_mode: bool = False):

    
    
    # Loss functions for training and validation.
    train_loss_func, eval_loss_func = loss.get_loss_funcs(args)
    
    # LR scheduler and epochs.
    w_optimizer = optim.get_optimizer(args, get_parameters(model=model, keys=['AP_path'], mode='exclude')) 
    arch_optimizer = torch.optim.Adam(get_parameters(model=model, keys=['AP_path_alpha'], mode='include'), args.alpha_lr, betas=(0.5, 0.999),
                                weight_decay=args.alpha_weight_decay)
    schedule = sche.get_schedule(args, w_optimizer)
    lr_scheduler, sched_epochs = schedule 
    
    # Using torch profiler to profile the training process
    if args.needs_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=4,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.canvas_tensorboard_log_dir, worker_name='worker0'),
            record_shapes=True,
            profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
            with_stack=True) 
        profiler.start()
    else:
        profiler = None
    
  # Create a logger.
    logger = log.get_logger()
    
    if args.canvas_tensorboard_log_dir:
        logger.info(f'Writing tensorboard logs to {args.canvas_tensorboard_log_dir}')
        writer = SummaryWriter(args.canvas_tensorboard_log_dir)
    else:
        writer = None
        
    if args.local_rank == 0:
        logger.info('Begin training ...')

    # AMP automatic cast TODO.
    amp_autocast, loss_scaler = suppress, None

    # Resume from checkpoint.
    resume_epoch = None
    if args.resume:
        if args.local_rank == 0:
            logger.info(f'Resuming from checkpoint {args.resume}')
        resume_epoch = resume_checkpoint(
            model, args.resume, optimizer=w_optimizer, loss_scaler=loss_scaler,
            log_info=(args.local_rank == 0)
        )
    start_epoch = resume_epoch if resume_epoch else 0
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # Checkpoint saver.TODO
    best_metric, best_epoch = None, None

    # Iterate over epochs.
    all_train_val_data_metrics = {}
    all_train_metrics, all_eval_metrics, magnitude_alphas, one_hot_alphas = [], [], [], []
    for epoch in range(start_epoch, sched_epochs):
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        
        # Pruner.TODO
        in_epoch_pruning_milestones = dict()

        # Train.
        train_metrics = train_one_epoch(args, epoch, model, train_loader, eval_loader, train_loss_func, eval_loss_func,
                                        w_optimizer, arch_optimizer, lr_scheduler, amp_autocast, loss_scaler, logger, writer, 
                                        pruning_milestones=in_epoch_pruning_milestones, profiler=profiler)
        all_train_metrics.append(train_metrics)
          
        if epoch == sched_epochs - 1:
            magnitude_alphas.append(get_sum_of_magnitude_scores_with_1D(model).tolist())
            
        # Log the parameters of the canvas kernels
        if epoch >= args.warmup_epochs:
            for i, placeholder in enumerate(model.canvas_cached_placeholders):
                placeholder.canvas_placeholder_kernel.print_parameters(i, epoch)

        # Check NaN errors.
        if math.isnan(train_metrics['loss']):
            raise RuntimeError('NaN occurs during training')
  
        # Evaluate.
        eval_metrics = validate(args, model, eval_loader, eval_loss_func, amp_autocast, logger)
        all_eval_metrics.append(eval_metrics)
        
        #  Log the parameters of the canvas kernels using tensorboard
        if writer:
            writer.add_scalar('Testing/Accuracy', eval_metrics['top1'], epoch)
            writer.add_scalar('Testing/Loss', eval_metrics['loss'], epoch)
        
        # Update LR scheduler.
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric])

        # Check NaN errors.
        if math.isnan(eval_metrics['loss']) and args.forbid_eval_nan:
            raise RuntimeError('NaN occurs during validation')

    if best_metric is not None:
        if args.local_rank == 0:
            logger.info(f'Best metric: {best_metric} (epoch {best_epoch})')
        
    # Stop the profiler
    if args.needs_profiler:
            profiler.stop()
            
    # Close the writer
    if writer:
        writer.close()

    return {
    "all_train_metrics": all_train_metrics,
    "all_eval_metrics": all_eval_metrics,
    "magnitude_alphas": magnitude_alphas,
    "one_hot_alphas": one_hot_alphas,
    "latest_top1": all_eval_metrics[-1]['top1']
    }

    