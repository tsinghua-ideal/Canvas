import math
import time
from collections import OrderedDict

import torch
import torch.profiler
from timm.models import model_parameters
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad

from . import log, loss, optim, sche
from .models import proxyless


def get_next_valid_sample(loader):
    while True:
        for sample in loader:
            yield sample


def train_one_epoch(args, epoch, model, train_loader, valid_loader, train_loss_func, valid_loss_func, model_optimizer,
                    arch_optimizer, lr_scheduler, logger):
    # Second order optimizer
    second_order = hasattr(model_optimizer, 'is_second_order') and model_optimizer.is_second_order

    # Meters
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    end = time.time()

    # Switch to training mode
    model.train()

    # Validation queue
    valid_queue = get_next_valid_sample(valid_loader)

    # Iterate over this epoch
    num_updates = epoch * len(train_loader)
    last_idx = len(train_loader) - 1
    for batch_idx, (image, target) in enumerate(train_loader):
        data_time_m.update(time.time() - end)

        # Random sample binary gates
        proxyless.sample_and_binarize(model, True)

        # Forward
        output = model(image)
        loss_value = train_loss_func(output, target)
        losses_m.update(loss_value.item(), image.size(0))

        model.zero_grad()

        # Backward
        loss_value.backward(create_graph=second_order)

        if args.clip_grad is not None:
            dispatch_clip_grad(
                model_parameters(model, exclude_head='agc' in args.clip_mode),
                value=args.clip_grad, mode=args.clip_mode)

        # Update model parameters only
        model_optimizer.step()

        # Restore modules
        proxyless.restore_modules(model)

        # Arch parameter updates
        if epoch > args.warmup_epochs:
            if (batch_idx + 1) % args.num_iters_update_alphas == 0:
                for i in range(args.alpha_update_steps):
                    valid_image, valid_target = next(valid_queue)
                    model.train()

                    # Make two of the kernels active and inactive
                    proxyless.sample_and_binarize(model, active_only=False)

                    # Forward
                    assert args.tta == 0
                    valid_output = model(valid_image)
                    loss = valid_loss_func(valid_output, valid_target)

                    # Backward and optimize
                    model.zero_grad()
                    loss.backward()
                    proxyless.set_alpha_grad(model)
                    arch_optimizer.step()
                    proxyless.rescale_alphas(model)

                    # Restore modules
                    proxyless.restore_modules(model)

        # Sync
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
            lrl = [param_group['lr'] for param_group in model_optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
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

        if math.isnan(losses_m.avg):
            break

    if hasattr(model_optimizer, 'sync_lookahead'):
        model_optimizer.sync_lookahead()

    metrics = OrderedDict([('loss', losses_m.avg)])
    return metrics


def validate(args, model, eval_loader, loss_func, logger):
    model.eval()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    end = time.time()
    last_idx = len(eval_loader) - 1

    # Only activate one module
    proxyless.sample_and_binarize(model, True)

    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(eval_loader):
            output = model(image)

            # Augmentation reduction
            assert args.tta == 0
            loss_value = loss_func(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.cuda.synchronize()
            losses_m.update(loss_value.item(), image.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            if math.isnan(losses_m.avg) and args.forbid_eval_nan:
                break

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging
            if batch_idx == last_idx or batch_idx % args.log_interval == 0:
                logger.info(
                    'Validate: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    # Restore modules
    proxyless.restore_modules(model)

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


def train(args, model, train_loader, eval_loader):
    # Loss functions for training and validation
    train_loss_func, valid_loss_func, eval_loss_func = loss.get_loss_funcs(args)

    # LR scheduler and epochs
    model_optimizer = optim.get_optimizer(args, proxyless.get_parameters(model=model, keys=['alphas'], mode='exclude'))
    arch_optimizer = torch.optim.Adam(proxyless.get_parameters(model=model, keys=['alphas'], mode='include'),
                                      args.alpha_lr, betas=(0.5, 0.999), weight_decay=args.alpha_weight_decay)
    schedule = sche.get_schedule(args, model_optimizer)
    lr_scheduler, sched_epochs = schedule

    # Create a logger
    logger = log.get_logger()

    # Asserts for proxyless trainer
    assert not args.resume
    assert not args.distributed

    # Iterate over epochs
    best_metric, best_epoch = None, None
    all_train_metrics, all_eval_metrics = [], []
    for epoch in range(1, sched_epochs + 1):
        # Train.
        train_metrics = train_one_epoch(args, epoch, model, train_loader, eval_loader, train_loss_func, valid_loss_func,
                                        model_optimizer, arch_optimizer, lr_scheduler, logger)
        all_train_metrics.append(train_metrics)

        # Log the parameters of the Canvas kernels
        # if epoch > args.warmup_epochs:
        #     proxyless.print_parameters(model)

        # Check NaN errors.
        if math.isnan(train_metrics['loss']):
            raise RuntimeError('NaN occurs during training')

        # Evaluate.
        eval_metrics = validate(args, model, eval_loader, eval_loss_func, logger)
        all_eval_metrics.append(eval_metrics)

        # Update LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(epoch, eval_metrics[args.eval_metric])

        # Check NaN errors
        if math.isnan(eval_metrics['loss']) and args.forbid_eval_nan:
            raise RuntimeError('NaN occurs during validation')

    if best_metric is not None:
        if args.local_rank == 0:
            logger.info(f'Best metric: {best_metric} (epoch {best_epoch})')

    return {
        'all_train_metrics': all_train_metrics,
        'all_eval_metrics': all_eval_metrics,
    }
