#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="resnet18"
# Dataset.
root="${current_dir}/datasets/imagenette/imagewoof2"
# root="${current_dir}/datasets/cifar10"

# Dataset augmentation.TO DO

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
lr=0.001
# Scheduler parameters.
epochs=80
cooldown_epochs=10
# Canvas preferences.
canvas_min_macs=0.01
canvas_max_macs=0.052
canvas_min_params=0.01
canvas_max_params=0.28
canvas_log_dir="${current_dir}/collections/baseline"
canvas_rounds=0
canvas_number_of_kernels=1
canvas_kernels="${current_dir}/collections/attn_lka.py ${current_dir}/collections/conv3x3.py"

# Lauch search_darts.py
CUDA_VISIBLE_DEVICES=2 python train.py --model $model --root $root --canvas-log-dir $canvas_log_dir --canvas-rounds $canvas_rounds --epochs $epochs \
 --canvas-min-macs $canvas_min_macs --canvas-max-macs $canvas_max_macs --canvas-min-params $canvas_min_params --canvas-max-params $canvas_max_params \
 --canvas-number-of-kernels $canvas_number_of_kernels --canvas-van-tiny --darts --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs


