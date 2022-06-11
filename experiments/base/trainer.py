import time
import torch

from timm.utils import AverageMeter


def train(args, model, train_loader, eval_loader, loss_func, optimizer, scheduler, sched_epochs):
    # CuDNN benchmarking.
    torch.backends.cudnn.benchmark = True

    # Iterate over epochs.
    for epoch in sched_epochs:
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
        for batch_idx, (image, target) in enumerate(train_loader):
            # Update starting time.
            data_time_m.update(time.time() - end)

            # Calculate loss.
            output = model(image)
            loss = loss_func(output, target)
            losses_m.update(loss.item(), image.size(0))
            optimizer.zero_grad()
            loss.backward(create_graph=second_order)
            optimizer.step()

            # Sync.
            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)

            # Update scheduler step.
            scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            # Update time.
            end = time.time()

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()
