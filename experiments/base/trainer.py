import math
import time
import torch
from collections import OrderedDict

from timm.models import model_parameters
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad, distribute_bn, reduce_tensor

from . import log, loss, optim, sche


def train_one_epoch(args, epoch, model, train_loader, loss_func, optimizer, lr_scheduler, logger):
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
    for batch_idx, (image, target) in enumerate(train_loader):
        # Update starting time.
        data_time_m.update(time.time() - end)

        # Calculate loss.
        output = model(image)
        loss_value = loss_func(output, target)

        if not args.distributed:
            losses_m.update(loss_value.item(), image.size(0))

        optimizer.zero_grad()
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

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(args, model, eval_loader, loss_func, logger):
    model.eval()

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    end = time.time()
    last_idx = len(eval_loader) - 1
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(eval_loader):
            output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Augmentation reduction.
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_func(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), image.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

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

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


def train(args, model, train_loader, eval_loader):
    # Loss functions for training and validation.
    train_loss_func, eval_loss_func = loss.get_loss_funcs(args)

    # LR scheduler and epochs.
    optimizer = optim.get_optimizer(args, model)
    schedule = sche.get_schedule(args, optimizer)
    lr_scheduler, sched_epochs = schedule

    # Create a logger.
    logger = log.get_logger()
    if args.local_rank == 0:
        logger.info('Begin training ...')

    # Iterate over epochs.
    all_train_metrics, all_eval_metrics = [], []
    for epoch in range(sched_epochs):
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Train.
        train_metrics = train_one_epoch(args, epoch, model, train_loader, train_loss_func,
                                        optimizer, lr_scheduler, logger)
        all_train_metrics.append(train_metrics)

        # Normalize.
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            # if args.local_rank == 0:
            #     logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        # Evaluate.
        eval_metrics = validate(args, model, eval_loader, eval_loss_func, logger)
        all_eval_metrics.append(eval_metrics)

        # Update LR scheduler.
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric])

        # Check NaN errors.
        if math.isnan(train_metrics['loss']) or math.isnan(eval_metrics['loss']):
            raise RuntimeError('NaN occurs during training')

    return all_train_metrics, all_eval_metrics
