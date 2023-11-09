#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="compact_van_b0"
num_classes=100
dataset="cifar100"

# Dataset.
root="${current_dir}/datasets/cifar100/"
dataset="cifar100"
seed=$((1 + $RANDOM % 100))
target_folder="${current_dir}/collections/preliminary_kernels_selected"
single_result_folder="${current_dir}/collections/validation_experiments/single_cifar100"
proportion_of_training_set=0.95

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
weight_decay=0.05
alpha_lr=0.001
alpha_weight_decay=0.0

clip_grad=0.01


# Scheduler parameters.
epochs=120
warmup_epochs=40
cooldown_epochs=0

# Canvas preferences.
canvas_number_of_kernels=40
canvas_log_dir="${current_dir}/collections/validation_experiments/final/${canvas_number_of_kernels}/multi_final"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/model_${model}_${opt}_lr_${lr}_wd_${weight_decay}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}_warmup_epoch_${warmup_epochs}"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python alpha_partial_order_validation_experiment.py \
 --seed $seed --alpha-weight-decay $alpha_weight_decay --alpha-lr $alpha_lr --dataset $dataset --root $root \
 --canvas-log-dir $canvas_log_dir  --epochs $epochs --num-classes $num_classes \
 --model $model --canvas-number-of-kernels $canvas_number_of_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs --proxyless \
 --needs-valid  --clip-grad $clip_grad  --search-mode --single-result-folder $single_result_folder --target-folder $target_folder \
 --canvas-tensorboard-log-dir $canvas_tensorboard_log_dir --proportion-of-training-set $proportion_of_training_set \
 --drop-path $drop_path --drop $drop --hflip $hflip --interpolation $interpolation --patch-size $patch_size \


 