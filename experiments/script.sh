#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="resnet18"
# Dataset.
root="${current_dir}/datasets/cifar10"
# Dataset augmentation.

# Loss functions.

# BatchNorm settings.

# Optimizer parameters.

# Scheduler parameters.
epochs=100
# Canvas preferences.
canvas_min_macs=0.01
canvas_max_macs=0.18
canvas_min_params=0.01
canvas_max_params=1.2
canvas_log_dir="${current_dir}/collections/test"
canvas_rounds=0
CUDA_VISIBLE_DEVICES=0 python search_darts.py --model $model --root $root --canvas-log-dir $canvas_log_dir --canvas-rounds $canvas_rounds --epochs $epochs \
 --canvas-min-macs $canvas_min_macs --canvas-max-macs $canvas_max_macs --canvas-min-params $canvas_min_params --canvas-max-params $canvas_max_params
