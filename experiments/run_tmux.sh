# #!/bin/bash
#!/bin/bash

num_windows=4

tmux new-session -d -s canvas

current_path=$(pwd)

for ((i = 0; i < num_windows; i++)); do
    tmux new-window -t canvas:$i
    tmux send-keys -t "$i" "bash" Enter
    tmux send-keys -t "$i" "zsh" Enter
    tmux send-keys -t "$i" "conda activate canvas" Enter
    tmux send-keys -t "$i" "cd ${current_path}" Enter

    # shellcheck disable=SC2004
    tmux send-keys -t "$i" "export CUDA_VISIBLE_DEVICES=$i" Enter
    
done

tmux select-window -t "$session_name:0"

tmux attach -t "canvas"



# # Run Canvas.
# current_path=$(pwd)
# machine_name=$(uname -a)
# for ((i = 0; i < num_windows; i ++)); do

# # You may change to your personal environment.
# tmux send-keys -t "$i" "bash" Enter
# tmux send-keys -t "$i" "zsh" Enter
# tmux send-keys -t "$i" "conda activate canvas" Enter
# tmux send-keys -t "$i" "cd ${current_path}" Enter
# # shellcheck disable=SC2004
# tmux send-keys -t "$i" "CUDA_VISIBLE_DEVICES=$(($i-1))" Enter
# done
# # Check power.
# if ! (( $1 > 0 && ($1 & ($1 - 1)) == 0 )); then
#   echo 'Process number should be a power of 2'
#   exit
# fi

# # Log2 function.
# function log2 {
#   local x=0
#   for ((y = $1 - 1; y > 0; y >>= 1)); do
#     ((x = x + 1))
#   done
#   echo "$x"
# }

# # Create windows.
# tmux new -s "Canvas" -d
# for ((i = 0; i < $(log2 "$1"); i ++)); do
#   for ((j = 0; j < 2 ** i; j ++)) do
#     tmux selectp -t $((j * 2 + 1))
#     tmux splitw -v -p 50
#   done
# done

# # Run Canvas.
# current_path=$(pwd)
# machine_name=$(uname -a)
# for ((i = 1; i <= $1; i ++)); do
# tmux send-keys -t "$i" "echo TMUX Pane $i" Enter
# # You may change to your personal environment.
# tmux send-keys -t "$i" "conda activate canvas" Enter
# tmux send-keys -t "$i" "cd ${current_path}" Enter
# # shellcheck disable=SC2004
# tmux send-keys -t "$i" "CUDA_VISIBLE_DEVICES=$(($i-1)) ${*:2}" Enter
# done

# # Attach.
# tmux attach -t "canvas"
