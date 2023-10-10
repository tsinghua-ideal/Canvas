#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="compact_van_b0"
# Dataset.
# root="${current_dir}/datasets/imagenette/imagewoof2"
# root="${current_dir}/datasets/cifar10/"
root="/dev/shm/shenao/cifar10/"

# Dataset augmentation.TO DO

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.003
weight_decay=0.0005
clip_grad=0.01

# Scheduler parameters.
epochs=35
warmup_epochs=5
cooldown_epochs=5
canvas_rounds=0
# Canvas preferences.
canvas_min_macs=0.01
canvas_max_macs=0.052
canvas_min_params=0.01
canvas_max_params=0.28
canvas_number_of_kernels=20
# validation_experiments/final"
canvas_log_dir="${current_dir}/collections/validation_experiments/final/20/ranking"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/${opt}_lr_${lr}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}_0.85_with_clip_gradp"

canvas_kernels="${current_dir}/collections/attn_lka.py ${current_dir}/collections/conv3x3.py"
# Launch the program.
# selector_experiment/alpha_partial_order_validation_experiment.py
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_single_kernel.py --model $model --root $root --canvas-log-dir $canvas_log_dir --canvas-rounds $canvas_rounds --epochs $epochs \
 --canvas-number-of-kernels $canvas_number_of_kernels --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs --use-multi-epochs-loader --clip-grad $clip_grad
#  --canvas-tensorboard-log-dir $canvas_tensorboard_log_dir 

