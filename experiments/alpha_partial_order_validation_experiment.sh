#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="compact_van_b0"
num_classes=100

# Dataset.
root="${current_dir}/datasets/cifar100/"

# Dataset augmentation.TO DO
mixup=0.0
cutmix=0.0
cutmix_minmax=None
smoothing=0.0
scale="0.95 1.0"
color_jitter=0.9
ratio="0.9 1.1111111111"
train_interpolation='bilinear'
interpolation='bilinear'
crop_pct=1.0

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.003
weight_decay=0.0005
alpha_lr=0.001
alpha_weight_decay=0.0
clip_grad=0.01

# Scheduler parameters.
epochs=70
warmup_epochs=30
cooldown_epochs=0
canvas_rounds=0

# Canvas preferences.
canvas_number_of_kernels=8
canvas_log_dir="${current_dir}/collections/validation_experiments/final/${canvas_number_of_kernels}/score_final"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/model_${model}_${opt}_lr_${lr}_wd_${weight_decay}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}_warmup_epoch_${warmup_epochs}"
canvas_kernels="${current_dir}/collections/attn_lka.py ${current_dir}/collections/conv3x3.py"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python alpha_partial_order_validation_experiment.py \
 --alpha-weight-decay $alpha_weight_decay --alpha-lr $alpha_lr --model $model --root $root \
 --canvas-log-dir $canvas_log_dir  --epochs $epochs --num-classes $num_classes \
 --canvas-number-of-kernels $canvas_number_of_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs --proxyless --use-multi-epochs-loader \
 --needs-valid --scale $scale --crop-pct $crop_pct --clip-grad $clip_grad --no-aug \
#   --smoothing $smoothing
#  --needs-profiler
# --train-interpolation $train_interpolation --interpolation $interpolation 
# --no-aug --crop-pct $crop_pct 
#  
#  --needs-profiler 

