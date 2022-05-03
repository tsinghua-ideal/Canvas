import canvas
import os
import ptflops
import time
import torch
from functools import partial
from pathlib import Path
from torch import nn, optim

from .pruner import Pruner
from .dataset import get_single_sample


supported_opt = {
    'SGD': partial(optim.SGD, momentum=0.9, weight_decay=5e-4, nesterov=True),
    'Adam': partial(optim.Adam),
    'AdamW': partial(optim.AdamW, weight_decay=5e-4)
}


def train(net: nn.Module,
          train_dataloader, test_dataloader=None,
          debug_iterations: int = 0, device: str = 'cuda',
          opt: str = 'SGD', scheduler: str = 'cos',
          epochs: int = 256, lr: float = 0.1,
          show_info: bool = False,
          show_replaced_info: bool = False,
          pruner: Pruner = None,
          save_checkpoint: str = ''):
    # Speed up
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Show complexity info
    # NOTE: Add hooks for unknown modules
    if show_info:
        sample_images, _ = get_single_sample(train_dataloader)
        sample_images = sample_images.to(device)
        macs, params = \
            ptflops.get_model_complexity_info(net, tuple(sample_images.shape),
                                              as_strings=False, print_per_layer_stat=False, verbose=False)
        print(' > MACs: {}'.format(macs))
        print(' > Params: {}'.format(params))

    # Show original convolution (before replaced) information
    if show_replaced_info:
        sample_images, _ = get_single_sample(train_dataloader, keep_dim=True)
        sample_images = sample_images.to(device)
        macs, params = canvas.statistics(net, sample_images, original=True, entire=False)
        print(' > Replaced MACs: {:.2f} GMac'.format(macs / 1e9))
        print(' > Replaced params: {:.2f} M'.format(params / 1e6))

    # Optimizer
    assert opt in supported_opt, f'Optimizer {opt} not supported in library'
    optimizer = supported_opt[opt](net.parameters(), lr=lr)

    # Scheduler
    if scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    else:
        print('No scheduler used')
        scheduler = None

    # Reset pruner
    if pruner is not None:
        pruner.reset()

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    # AMP (only for CUDA)
    assert device == 'cuda'
    scaler = torch.cuda.amp.GradScaler()

    # Training
    details = {'score': 0, 'epochs': 0, 'acc': []}
    steps, score = 0, 0
    train_startup_timestamp = time.time()
    for i in range(epochs):
        pruned = False

        # Train
        net.train()
        n_iter, loss_sum, epoch_steps = 0, 0, 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outs = net(images)
                loss = criterion(outs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            epoch_steps += 1
            loss_sum += loss.detach().item()
            print(f'\33[2K\r > Loss@[{steps}, {i + 1}/{epochs}]: {loss_sum / epoch_steps}', end='', flush=True)
            n_iter += 1
            if pruner is not None:
                pruned |= pruner.prune_loss(loss_sum)
            if pruned or (debug_iterations != 0 and n_iter == debug_iterations):
                break
        print()
        details['epochs'] += 1

        # Scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate
        if not pruned and test_dataloader is not None:
            print(' > Evaluating ...', end='', flush=True)
            net.eval()
            num_samples, num_corrects = 0, 0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outs = net(images)
                    _, predicted = outs.max(1)
                    num_samples += images.size(0)
                    num_corrects += predicted.eq(labels).sum().item()
            score = max(score, num_corrects / num_samples)
            details['score'] = score
            details['acc'].append(num_corrects / num_samples)
            print(f'\33[2K\r > Accuracy@{i + 1}: '
                  f'{num_corrects / num_samples} ({num_corrects}/{num_samples}, max: {score})', end='', flush=True)
            print()
            if pruner is not None:
                assert pruner is not None
                pruned |= pruner.update_epoch(score, num_corrects / num_samples)

        # Early stopping
        if pruned:
            assert len(pruner.current_pruned_info) > 0
            print(f' $ Pruned: {pruner.current_pruned_info}')
            break

    # Save timestamp
    details['training_time'] = time.time() - train_startup_timestamp

    if save_checkpoint:
        print(f' > Saving checkpoint into {save_checkpoint} ...')
        parent_path = Path(save_checkpoint).parent.absolute()
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        torch.save(net.state_dict(), save_checkpoint)

    if 'score' not in details:
        details['score'] = 0
    return details
