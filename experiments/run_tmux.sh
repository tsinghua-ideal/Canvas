#!/bin/bash

current_path=$(pwd)

# Create windows.
tmux new -s "Canvas" -d
tmux selectp -t 1
tmux splitw -v -p 50
tmux selectp -t 1
tmux splitw -v -p 50
tmux selectp -t 3
tmux splitw -v -p 50

# Run Canvas.
for i in {1..4}
do
tmux send-keys -t "$i" "echo TMUX Pane $i" Enter
# You may change to your personal environment.
tmux send-keys -t "$i" "source ~/.local/miniconda3/etc/profile.d/conda.sh; conda activate Canvas" Enter
tmux send-keys -t "$i" "cd ${current_path}" Enter
# shellcheck disable=SC2004
tmux send-keys -t "$i" "CUDA_VISIBLE_DEVICES=$(($i-1)) $*" Enter
done

# Attach.
tmux attach -t "Canvas"
