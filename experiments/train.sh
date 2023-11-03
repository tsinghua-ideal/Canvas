#!/bin/bash
# Misc.
current_dir=$(dirname "$0")

# Models.
model="compact_van_b0"
num_classes=100
dataset="cifar100"

# Dataset.
root="/dev/shm/shenao/cifar100/"
seed=42

# Dataset augmentation.TO DO

# Loss functions.TO DO

# BatchNorm settings.TO DO

# Optimizer parameters.TO DO
opt="adamw"
lr=0.003
weight_decay=0.05
clip_grad=0.01

# Scheduler parameters.
epochs=90
warmup_epochs=30
cooldown_epochs=0

# Canvas preferences.
canvas_kernels="${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py
${current_dir}/collections/nice_kernels/conv7x7.py ${current_dir}/collections/nice_kernels/conv3x3.py" 
canvas_log_dir="${current_dir}/collections/baseline/cifar100/one"
canvas_tensorboard_log_dir="${canvas_log_dir}/tensorboard/model_${model}_kernel_num_classes_${num_classes}_${opt}_lr_${lr}_weight_decay_${weight_decay}_alpha_lr_${alpha_lr}_num_${canvas_number_of_kernels}_machine_${HOSTNAME}_GPU_${CUDA_VISIBLE_DEVICES}"

# Launch the program.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py  --model $model --dataset $dataset --seed $seed --root $root --canvas-log-dir $canvas_log_dir --epochs $epochs \
  --canvas-kernels $canvas_kernels --lr $lr --cooldown-epochs $cooldown_epochs \
 --opt $opt --weight-decay $weight_decay  --warmup-epochs $warmup_epochs  --use-multi-epochs-loader \
 --canvas-tensorboard-log-dir $canvas_tensorboard_log_dir --num-classes $num_classes  --clip-grad $clip_grad \
 --crop-pct $crop_pct  --scale $scale --needs-valid --proxyless