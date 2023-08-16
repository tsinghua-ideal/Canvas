

import json
import math
import os
import time
import torch
import canvas

from torch.autograd import grad
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.models import model_parameters, resume_checkpoint, safe_model_name
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad, distribute_bn, reduce_tensor
from timm.utils import NativeScaler, ApexScaler, CheckpointSaver, get_outdir, update_summary

from . import log, loss, optim, sche, darts


def train_one_epoch(args, epoch, model, train_loader, 
                    loss_func, optimizer, lr_scheduler,
                    amp_autocast, loss_scaler, logger, evaluate,
                    pruning_milestones, w_optim = None, alpha_optim = None):
    # Second order optimizer.
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    # Meters.
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()
    num_updates = epoch * len(train_loader)

    # Switch to training mode.
    model.train()

    # Iterate over this epoch.
    last_idx = len(train_loader) - 1
    for batch_idx, (image, target)in enumerate(train_loader):
        # lr = lr_scheduler._get_lr(epoch)[0]
        # Update starting time.
        data_time_m.update(time.time() - end)

        # Calculate loss.
        # alpha_optim.zero_grad()
        # Architect step
        # architect.unrolled_backward(image, target, val_image, val_target, lr, w_optim)
        # Weights step
       
        
        with amp_autocast():
            output = model(image)
            loss_value = loss_func(output, target)

        if not args.distributed:
            losses_m.update(loss_value.item(), image.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss_value, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss_value.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
            
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
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

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

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = OrderedDict([('loss', losses_m.avg)])
    if args.darts and evaluate:
        alphas = {}
        for parallel_kernels in canvas.get_placeholders(model):
            assert isinstance(parallel_kernels.canvas_placeholder_kernel, darts.ParallelKernels)
            alphas[f'In the {epoch} epoch:'] = darts.get_alphas(model, detach = True)
        # if args.local_rank == 0:
        #     logger.info(f'Alphas: {alphas}')
        metrics.update({'alphas': alphas})
    return metrics


def validate(args, model, eval_loader, loss_func, amp_autocast, logger):
    model.eval()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    end = time.time()
    last_idx = len(eval_loader) - 1
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

    # Record kernel scales.
    kernel_scales = []
    if hasattr(model, 'kernel_scales'):
        kernel_scales = model.kernel_scales()
        logger.info(f'Kernel scales: {kernel_scales}')

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg),
                        ('top5', top5_m.avg), ('kernel_scales', kernel_scales)])


def train(args, model, train_loader, eval_loader, search_mode: bool = False, proxy_mode: bool = False, evaluate = False):
    # Set different number of epochs based on your target
        
    # Loss functions for training and validation.
    train_loss_func, eval_loss_func = loss.get_loss_funcs(args)
    
    # LR scheduler and epochs.
    optimizer = optim.get_optimizer(args, model)
    schedule = sche.get_schedule(args, optimizer)
    lr_scheduler, sched_epochs = schedule 
    w_optim = torch.optim.SGD(darts.get_weights(model), args.w_lr, momentum=args.w_momentum,
                              weight_decay=args.w_weight_decay)
    # alpha_optim = torch.optim.Adam(darts.get_alphas(model), args.alpha_lr, betas=(0.5, 0.999),
    #                                weight_decay=args.alpha_weight_decay)
    alpha_optim = None
    # Create a logger.
    logger = log.get_logger()
    
    if args.local_rank == 0:
        logger.info('Begin training ...')

    # AMP automatic cast.
    if args.native_amp:
        if args.local_rank == 0:
            logger.info('Training with native PyTorch AMP')
        amp_autocast, loss_scaler = torch.cuda.amp.autocast, NativeScaler()
    elif args.apex_amp:
        if args.local_rank == 0:
            logger.info(f'Training with native Apex AMP (loss scale: {args.apex_amp_loss_scale})')
        amp_autocast, loss_scaler = suppress, ApexScaler()
        # noinspection PyUnresolvedReferences
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0,
                                          loss_scale=args.apex_amp_loss_scale)
    else:
        amp_autocast, loss_scaler = suppress, None

    # Resume from checkpoint.
    resume_epoch = None
    if args.resume:
        if args.local_rank == 0:
            logger.info(f'Resuming from checkpoint {args.resume}')
        resume_epoch = resume_checkpoint(
            model, args.resume, optimizer=optimizer, loss_scaler=loss_scaler,
            log_info=(args.local_rank == 0)
        )
    start_epoch = resume_epoch if resume_epoch else 0
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # Distributed training.
    if args.distributed:
        if args.local_rank == 0:
            logger.info("Using native Torch DistributedDataParallel.")
        if args.apex_amp:
            # noinspection PyUnresolvedReferences
            from apex.parallel import DistributedDataParallel as ApexDDP
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)

    # Checkpoint saver.
    best_metric, best_epoch = None, None
    saver, output_dir = None, None
    if args.local_rank == 0 and not search_mode:
        name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model)
        ])
        output_dir = get_outdir(args.output if args.output else './output/train', name)
        decreasing = True if args.eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.json'), 'w') as file:
            json.dump(vars(args), fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)

    # Pruning after epochs.
    # overall_pruning_milestones = None
    # if args.canvas_epoch_pruning_milestone:
    #     with open(args.canvas_epoch_pruning_milestone) as f:
    #         overall_pruning_milestones = json.load(f)
    #     if args.local_rank == 0:
    #         logger.info(f'Milestones (overall epochs) loaded: {overall_pruning_milestones}')

    # Iterate over epochs.
    all_train_metrics, all_eval_metrics = [], []
    for epoch in range(start_epoch, sched_epochs):
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # # Pruner.
        in_epoch_pruning_milestones = dict()
        # if epoch == 0 and search_mode and not proxy_mode and args.canvas_first_epoch_pruning_milestone:
        #     with open(args.canvas_first_epoch_pruning_milestone) as f:
        #         in_epoch_pruning_milestones = json.load(f)
        #     if args.local_rank == 0:
        #         logger.info(f'Milestones (first-epoch loss) loaded: {in_epoch_pruning_milestones}')

        # Train.
        train_metrics = train_one_epoch(args, epoch, model, train_loader, train_loss_func,
                                        optimizer, lr_scheduler, amp_autocast, loss_scaler, logger, evaluate,
                                        pruning_milestones=in_epoch_pruning_milestones)
        all_train_metrics.append(train_metrics)
        
        # Log the parameters
        if evaluate:
            for placeholder in model.canvas_cached_placeholders:
                placeholder.canvas_placeholder_kernel.print_parameters(epoch)

        # Check NaN errors.
        if math.isnan(train_metrics['loss']):
            raise RuntimeError('NaN occurs during training')

        # Normalize.
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            # if args.local_rank == 0:
            #     logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            
        # Evaluate.
        eval_metrics = validate(args, model, eval_loader, eval_loss_func, amp_autocast, logger)
        all_eval_metrics.append(eval_metrics)

        # Update LR scheduler.
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric])

        # Check NaN errors.
        if math.isnan(eval_metrics['loss']) and args.forbid_eval_nan:
            raise RuntimeError('NaN occurs during validation')

        # Summary and save checkpoint.
        if output_dir is not None:
            if args.local_rank == 0:
                logger.info('Updating summary ...')
            update_summary(
                epoch, train_metrics, eval_metrics,
                os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

    if best_metric is not None:
        if args.local_rank == 0:
            logger.info(f'Best metric: {best_metric} (epoch {best_epoch})')
    return all_train_metrics, all_eval_metrics
