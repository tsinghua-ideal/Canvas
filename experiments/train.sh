#!/bin/bash
# Misc.
current_dir=$(dirname "$0")
output="${current_dir}/output/compact_van_cifar10"

# Models.
model="compact_van_b0"

# Dataset.
root="/dev/shm/shenao/cifar10/"

# Dataset augmentation.TO DO

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.003
weight_decay=0.005
alpha_lr=0.001
alpha_weight_decay=0.0
clip_grad=0.01

# Scheduler parameters.
epochs=75
warmup_epochs=15
cooldown_epochs=0
canvas_rounds=0

# Canvas preferences.
canvas_min_macs=0.01
canvas_max_macs=0.052
canvas_min_params=0.01
canvas_max_params=0.28
canvas_number_of_kernels=5
canvas_log_dir="${current_dir}/collections/baseline/cifar10/pretrained"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/${opt}_lr_${lr}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}_0.85_with_clip_grad"
canvas_kernels="${current_dir}/collections/nice_kernels/identity.py"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py --alpha-weight-decay $alpha_weight_decay --alpha-lr $alpha_lr --model $model --root $root --canvas-log-dir $canvas_log_dir --canvas-rounds $canvas_rounds --epochs $epochs \
 --canvas-number-of-kernels $canvas_number_of_kernels --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs --proxyless --use-multi-epochs-loader --clip-grad $clip_grad  --output $output  
#  --canvas-tensorboard-log-dir $canvas_tensorboard_log_dir --needs-profiler

