#!/bin/bash


# Misc.
current_dir=$(dirname "$0")

# Models.
model="resnet18"
num_classes=100

# Dataset.
root="${current_dir}/datasets/cifar100/"
seed=42
dataset="cifar100"
proportion_of_training_set=0.95
batch_size=128

# Dataset augmentation.TO DO
drop=0.125
drop_path=0.125
patch_size=8
vflip=0.5
hflip=0.0
interpolation='bilinear'

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.006
weight_decay=0.075
clip_grad=0.01

# Scheduler parameters.
epochs=1
warmup_epochs=0
cooldown_epochs=0

# Canvas preferences.
needs_replace=False

canvas_kernels="${current_dir}/collections/nice_kernels/conv1x1.py"
canvas_log_dir="${current_dir}/collections/baseline/cifar100/51/${opt}_lr_${lr}_weight_decay_${weight_decay}_alpha_lr_${alpha_lr}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/model_${model}_kernel_num_classes_${num_classes}_${opt}_lr_${lr}_weight_decay_${weight_decay}_alpha_lr_${alpha_lr}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py  --model $model  --canvas-log-dir $canvas_log_dir \
    --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs --seed $seed --root $root \
    --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs --dataset $dataset \
    --num-classes $num_classes --epochs $epochs --clip-grad $clip_grad \
    --drop-path $drop_path --drop $drop \
    --patch-size $patch_size --hflip $hflip --interpolation $interpolation \
    # --proportion-of-training-set $proportion_of_training_set  --proxyless --needs-valid \
    # --needs-replace $needs_replace