source /home3/asrdictt/taoyu/bashrc_mxnet
#### Kill program on remote machines
python dmlc-submit --cluster mpi --num-servers 16 --num-workers 16 sh ./mlp/run_kill_cmd.sh
#### Kill program on local machine
sh ./mlp/run_kill_cmd.sh
