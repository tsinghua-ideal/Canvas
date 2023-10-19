#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="compact_van_b0"
num_classes=10

# Dataset.
root="/dev/shm/shenao/cifar10/"

# Dataset augmentation.TO DO
mixup=0.0
cutmix=0.0
cutmix_minmax=None
smoothing=0.0
scale="0.95 1.0"
color_jitter=0.8
ratio="0.9 1.1111111111"
train_interpolation='bilinear'
interpolation='bilinear'
crop_pct=1.0

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.003
weight_decay=0.05
clip_grad=0.01

# Scheduler parameters.
epochs=105
warmup_epochs=5
cooldown_epochs=0

# Canvas preferences.
# canvas_kernels="${current_dir}/collections/nice_kernels/conv3x3.py"
canvas_kernels="${current_dir}/collections/nice_kernels/identity.py"
canvas_log_dir="${current_dir}/collections/baseline/cifar10/one"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/model_${model}_kernel_${canvas_kernels}_num_classes_${num_classes}_${opt}_lr_${lr}_weight_decay_${weight_decay}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py  --model $model  --root $root --canvas-log-dir $canvas_log_dir --epochs $epochs \
  --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs  --use-multi-epochs-loader \
 --canvas-tensorboard-log-dir $canvas_tensorboard_log_dir --num-classes $num_classes  --clip-grad $clip_grad \
 --scale $scale --crop-pct $crop_pct 
#   --smoothing $smoothing
#  --needs-profiler
# --train-interpolation $train_interpolation --interpolation $interpolation 
# --no-aug --crop-pct $crop_pct 